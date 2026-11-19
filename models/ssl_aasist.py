"""
Upgrade Model: SSL (wav2vec2 / XLS-R) Front End + AASIST Graph Attention Back End + Dual Heads + GRL.
Adapted from TakHemlata/SSL_Anti-spoofing and clovaai/aasist.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grl import GradientReversalLayer


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_uniform_(self.att_weight)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, nodes, dim)
        x = self.input_drop(x)
        proj = self.att_proj(x)
        # Attention map: (bs, nodes, nodes)
        score = torch.matmul(proj, self.att_weight).squeeze(-1)  # (bs, nodes)
        att_map = torch.sigmoid(score / self.temp).unsqueeze(-1)  # (bs, nodes, 1)

        out = self.proj_with_att(x) * att_map + self.proj_without_att(x)
        bs, nodes, dim = out.shape
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        return out


class SSLFeatureExtractor(nn.Module):
    """
    Wrapper for SSL models (wav2vec 2.0 / XLS-R) via HuggingFace transformers,
    with fallback to a 1D conv waveform encoder if offline or mock.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base", freeze_ssl: bool = True, mock_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        self.freeze_ssl = freeze_ssl
        self.ssl_model = None
        self.out_dim = mock_dim
        self.is_hf = False

        if model_name.lower() in ["native", "mock", "fallback", "conv1d"]:
            self.fallback_conv = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, mock_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(mock_dim),
                nn.GELU()
            )
            self.is_hf = False
            return

        try:
            from transformers import AutoModel
            print(f"Loading SSL model: {model_name}...")
            self.ssl_model = AutoModel.from_pretrained(model_name)
            if freeze_ssl:
                for param in self.ssl_model.parameters():
                    param.requires_grad = False
            self.out_dim = self.ssl_model.config.hidden_size
            self.is_hf = True
            print(f"SSL model loaded successfully (out_dim={self.out_dim}).")
        except Exception as e:
            print(f"Notice: Could not load HuggingFace SSL model ({e}). Using native Sinc/Conv1D encoder fallback.")
            # Lightweight 1D waveform feature extractor matching wav2vec 2.0 output dims
            self.fallback_conv = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, mock_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(mock_dim),
                nn.GELU()
            )
            self.is_hf = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, L)
        if self.is_hf:
            with torch.set_grad_enabled(not self.freeze_ssl):
                outputs = self.ssl_model(wav)
                return outputs.last_hidden_state  # (B, T, D)
        else:
            if wav.ndim == 2:
                wav = wav.unsqueeze(1)
            feats = self.fallback_conv(wav)  # (B, D, T)
            return feats.transpose(1, 2)  # (B, T, D)


class SSLAASISTModel(nn.Module):
    def __init__(self,
                 ssl_model_name: str = "facebook/wav2vec2-base",
                 freeze_ssl: bool = True,
                 embedding_dim: int = 128,
                 gat_hidden: int = 64,
                 dropout: float = 0.3,
                 use_grl: bool = True,
                 lambda_val: float = 1.0,
                 num_emotions: int = 5):
        super().__init__()
        self.ssl_extractor = SSLFeatureExtractor(model_name=ssl_model_name, freeze_ssl=freeze_ssl)
        ssl_dim = self.ssl_extractor.out_dim
        self.embedding_dim = embedding_dim
        self.use_grl = use_grl

        # Project SSL features into graph dimension
        self.in_proj = nn.Linear(ssl_dim, gat_hidden)

        # AASIST Graph Attention Layers
        self.gat1 = GraphAttentionLayer(gat_hidden, gat_hidden)
        self.gat2 = GraphAttentionLayer(gat_hidden, gat_hidden)

        # Graph pooling & embedding projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_z = nn.Sequential(
            nn.Linear(gat_hidden, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        # Spoof Head (2 classes)
        self.spoof_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2)
        )

        # Emotion Head (5 classes via GRL)
        self.grl = GradientReversalLayer(lambda_val=lambda_val)
        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_emotions)
        )

    def set_lambda(self, lambda_val: float):
        self.grl.set_lambda(lambda_val)

    def extract_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        ssl_feats = self.ssl_extractor(wav)  # (B, T, D)
        h = self.in_proj(ssl_feats)          # (B, T, gat_hidden)
        h = self.gat1(h)                     # (B, T, gat_hidden)
        h = self.gat2(h)                     # (B, T, gat_hidden)
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # (B, gat_hidden)
        z = self.fc_z(pooled)                # (B, embedding_dim)
        return z

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.extract_embedding(wav)
        spoof_logits = self.spoof_head(z)
        if self.use_grl:
            z_emo = self.grl(z)
        else:
            z_emo = z
        emotion_logits = self.emotion_head(z_emo)
        return z, spoof_logits, emotion_logits
