"""
Proposed Model: Hybrid Feature CNN-BiLSTM with Dual Heads and GRL.
Features: MFCC + log-Mel + deltas fused via concatenation or parallel conv branches.
Shared Encoder -> embedding z -> Spoof Head (2-way) + Adversarial Emotion Head (5-way via GRL).
Includes fast on-GPU front-end extraction from raw waveforms (110x speedup).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from models.grl import GradientReversalLayer


class SelfAttentionPooling(nn.Module):
    """Self-attentive temporal pooling to map variable sequence length to fixed-size vector."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.att_linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time_steps, in_dim)
        weights = torch.softmax(self.att_linear(x), dim=1)  # (batch_size, time_steps, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, in_dim)
        return pooled


class ParallelConvFusion(nn.Module):
    """Parallel convolutional branches for Mel and MFCC/deltas before fusing."""
    def __init__(self, out_channels: int = 32):
        super().__init__()
        # Branch 1 for log-Mel (1 channel, 80 mel bins)
        self.mel_branch = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Branch 2 for MFCC + Deltas (2 channels, 40 bins -> resized or interpolated)
        self.mfcc_branch = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, mel: torch.Tensor, mfcc_delta: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, 80, T), mfcc_delta: (B, 2, 40, T)
        h_mel = self.mel_branch(mel)  # (B, out/2, 40, T/2)
        h_mfcc = self.mfcc_branch(mfcc_delta)  # (B, out/2, 20, T/2)
        # Interpolate frequency dimension of mfcc branch to match mel
        h_mfcc = F.interpolate(h_mfcc, size=(h_mel.shape[2], h_mel.shape[3]), mode="bilinear", align_corners=False)
        # Fuse by concatenating channels
        fused = torch.cat([h_mel, h_mfcc], dim=1)  # (B, out_channels, 40, T/2)
        return fused


class ProposedCNNBiLSTM(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 fusion_strategy: str = "concat",  # 'concat' or 'parallel_conv'
                 embedding_dim: int = 128,
                 lstm_hidden: int = 128,
                 dropout: float = 0.3,
                 use_grl: bool = True,
                 lambda_val: float = 1.0,
                 num_emotions: int = 5,
                 feature_type: str = "combined"):
        super().__init__()
        self.in_channels = in_channels
        self.fusion_strategy = fusion_strategy
        self.embedding_dim = embedding_dim
        self.use_grl = use_grl
        self.feature_type = feature_type

        # On-GPU acoustic front-end transforms (applied if input is raw waveform B x T)
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, power=2.0
        )
        self.mfcc_extractor = T.MFCC(
            sample_rate=16000, n_mfcc=40,
            melkwargs={"n_fft": 512, "n_mels": 80, "win_length": 400, "hop_length": 160, "mel_scale": "htk"}
        )
        self.delta_extractor = T.ComputeDeltas()

        # 1. Convolutional Front End
        if fusion_strategy == "parallel_conv":
            self.parallel_fusion = ParallelConvFusion(out_channels=32)
            conv_in = 32
        else:
            self.parallel_fusion = None
            conv_in = in_channels

        self.conv_net = nn.Sequential(
            nn.Conv2d(conv_in, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout * 0.5),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout * 0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout)
        )

        # 2. Adaptive frequency pooling to decouple frequency dimension from CNN input resolution
        self.freq_pool = nn.AdaptiveAvgPool2d((8, None))
        conv_out_features = 128 * 8

        self.bilstm = nn.LSTM(
            input_size=conv_out_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )

        # 3. Temporal Pooling -> embedding z
        self.pooling = SelfAttentionPooling(in_dim=lstm_hidden * 2)
        self.fc_z = nn.Sequential(
            nn.Linear(lstm_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        # 4. Spoof Detection Head (2 classes: bona_fide, spoof)
        self.spoof_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2)
        )

        # 5. Emotion Classification Head (5 classes: neutral, happy, angry, sad, surprise)
        self.grl = GradientReversalLayer(lambda_val=lambda_val)
        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_emotions)
        )

    def set_lambda(self, lambda_val: float):
        """Update GRL lambda coefficient dynamically."""
        self.grl.set_lambda(lambda_val)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to embedding z."""
        if x.ndim == 2:
            # Fast on-GPU acoustic front-end from raw waveforms (B, T)
            mel = self.mel_extractor(x)
            log_mel = torch.log(torch.clamp(mel, min=1e-6))
            
            if self.in_channels == 1:
                if self.feature_type == "mfcc_only":
                    mfcc = self.mfcc_extractor(x)
                    x = mfcc.unsqueeze(1)
                else:
                    x = log_mel.unsqueeze(1)
            else:
                mfcc = self.mfcc_extractor(x)
                delta = self.delta_extractor(mfcc)
                # Pad MFCC and Delta from 40 to 80 bins
                mfcc_pad = F.pad(mfcc, (0, 0, 0, 40))
                delta_pad = F.pad(delta, (0, 0, 0, 40))
                x = torch.stack([log_mel, mfcc_pad, delta_pad], dim=1)

        if self.fusion_strategy == "parallel_conv" and x.shape[1] == 3:
            mel = x[:, 0:1, :, :]
            mfcc_delta = x[:, 1:3, :40, :]
            h = self.parallel_fusion(mel, mfcc_delta)
        else:
            h = x

        h = self.conv_net(h)  # (B, 128, F', T')
        h = self.freq_pool(h)  # Guarantee frequency is exactly 8
        B, C, F_dim, T_dim = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)  # (B, T', C*8)

        lstm_out, _ = self.bilstm(h)  # (B, T', 2 * hidden)
        pooled = self.pooling(lstm_out)  # (B, 2 * hidden)
        z = self.fc_z(pooled)  # (B, embedding_dim)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: Shared embedding (B, embedding_dim)
            spoof_logits: Binary classification logits (B, 2)
            emotion_logits: 5-way emotion classification logits (B, 5)
        """
        z = self.extract_embedding(x)
        spoof_logits = self.spoof_head(z)

        if self.use_grl:
            z_emo = self.grl(z)
        else:
            z_emo = z  # No GRL ablation

        emotion_logits = self.emotion_head(z_emo)
        return z, spoof_logits, emotion_logits
