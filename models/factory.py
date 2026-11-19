"""
Model factory for instantiating anti-spoofing architectures.
Supported architectures:
- 'proposed_cnn_bilstm': Feature-based CNN-BiLSTM with dual heads and GRL
- 'ssl_aasist': wav2vec2/XLS-R + AASIST GAT backend with dual heads and GRL
- 'aasist': Pretrained or retrained AASIST (clovaai/aasist)
- 'rawnet2': RawNet2 waveform baseline
"""

from typing import Any, Dict
import torch
import torch.nn as nn
from pathlib import Path

from models.proposed_cnn_bilstm import ProposedCNNBiLSTM
from models.ssl_aasist import SSLAASISTModel
from models.aasist import Model as AASISTModel
from models.rawnet2 import Model as RawNet2Model


def create_model(model_name: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name in ["proposed", "proposed_cnn_bilstm", "cnn_bilstm"]:
        m_cfg = config.get("model", {})
        d_cfg = config.get("data", {})
        model = ProposedCNNBiLSTM(
            in_channels=m_cfg.get("in_channels", 3),
            fusion_strategy=m_cfg.get("fusion_strategy", "concat"),
            embedding_dim=m_cfg.get("embedding_dim", 128),
            lstm_hidden=m_cfg.get("lstm_hidden", 128),
            dropout=m_cfg.get("dropout", 0.3),
            use_grl=m_cfg.get("use_grl", True),
            lambda_val=m_cfg.get("lambda_val", 1.0),
            num_emotions=m_cfg.get("num_emotions", 5),
            feature_type=d_cfg.get("feature_type", "combined")
        )
    elif model_name in ["ssl", "ssl_aasist", "wav2vec2_aasist", "xlsr_aasist"]:
        m_cfg = config.get("model", {})
        model = SSLAASISTModel(
            ssl_model_name=m_cfg.get("ssl_model_name", "facebook/wav2vec2-base"),
            freeze_ssl=m_cfg.get("freeze_ssl", True),
            embedding_dim=m_cfg.get("embedding_dim", 128),
            gat_hidden=m_cfg.get("gat_hidden", 64),
            dropout=m_cfg.get("dropout", 0.3),
            use_grl=m_cfg.get("use_grl", True),
            lambda_val=m_cfg.get("lambda_val", 1.0),
            num_emotions=m_cfg.get("num_emotions", 5)
        )
    elif model_name in ["aasist", "clovaai_aasist"]:
        # Standard AASIST baseline
        default_aasist_cfg = {
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        m_cfg = config.get("model_config") or default_aasist_cfg
        model = AASISTModel(m_cfg)
        pretrained_path = config.get("pretrained_weights", None)
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading official AASIST pretrained weights from {pretrained_path}...")
            state = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state)
            print("Loaded AASIST pretrained weights successfully.")
    elif model_name in ["rawnet2", "rawnet"]:
        default_rawnet_cfg = {
            "nb_samp": 64600,
            "first_conv": 1024,
            "in_channels": 1,
            "filts": [20, [20, 20], [20, 128], [128, 128]],
            "blocks": [2, 4],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 3,
            "nb_classes": 2
        }
        m_cfg = config.get("model_config") or default_rawnet_cfg
        model = RawNet2Model(m_cfg)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    return model.to(device)
