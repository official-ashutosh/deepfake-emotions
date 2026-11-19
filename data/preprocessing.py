"""
Audio preprocessing and feature extraction module.
Standardizes audio to 16 kHz, peak normalized, fixed length (~4s, 64000 samples).
Extracts MFCC, log-Mel spectrogram, delta, and delta-delta features.
Supports computing and applying dataset-level normalization stats strictly from train set.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path


class AudioPreprocessor:
    def __init__(self,
                 target_sr: int = 16000,
                 target_samples: int = 64000, # 4.0 seconds at 16 kHz
                 n_fft: int = 512,
                 win_length: int = 400, # 25 ms
                 hop_length: int = 160, # 10 ms
                 n_mels: int = 80,
                 n_mfcc: int = 40):
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        # Torchaudio feature extractors
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        self.mfcc_transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "win_length": win_length,
                "hop_length": hop_length,
                "mel_scale": "htk"
            }
        )
        self.delta_transform = T.ComputeDeltas()

    def load_audio(self, filepath: Union[str, Path], is_train: bool = False) -> torch.Tensor:
        """
        Load audio file, convert to mono, resample to 16 kHz, peak normalize,
        and crop/pad to target_samples with identical silence handling.
        """
        filepath = str(filepath)
        wav_np, sr = sf.read(filepath, dtype="float32")
        
        # Convert to mono if multi-channel
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=1)

        wav = torch.from_numpy(wav_np)

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = resampler(wav.unsqueeze(0)).squeeze(0)

        # Peak normalization
        peak = torch.max(torch.abs(wav))
        if peak > 1e-7:
            wav = wav / peak

        # Fixed length handling (~4 s, 64000 samples)
        length = wav.shape[0]
        if length < self.target_samples:
            # Repeat or pad
            repeat_count = int(np.ceil(self.target_samples / length))
            wav = wav.repeat(repeat_count)[:self.target_samples]
        elif length > self.target_samples:
            if is_train:
                # Random crop during training
                max_start = length - self.target_samples
                start = torch.randint(0, max_start + 1, (1,)).item()
                wav = wav[start:start + self.target_samples]
            else:
                # Deterministic center crop during dev/test
                start = (length - self.target_samples) // 2
                wav = wav[start:start + self.target_samples]

        return wav

    def extract_features(self, wav: torch.Tensor,
                         include_deltas: bool = True,
                         include_delta_deltas: bool = False,
                         norm_stats: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract log-Mel, MFCC, and delta features from (64000,) waveform tensor.
        Returns dictionary of tensors.
        """
        if wav.ndim == 1:
            wav_unsq = wav.unsqueeze(0)
        else:
            wav_unsq = wav

        # 1. Log-Mel spectrogram (F=80, T)
        mel = self.mel_spectrogram(wav_unsq)
        log_mel = torch.log(torch.clamp(mel, min=1e-6))
        
        # 2. MFCC (F=40, T)
        mfcc = self.mfcc_transform(wav_unsq)

        # 3. Deltas
        feats = {"log_mel": log_mel.squeeze(0), "mfcc": mfcc.squeeze(0)}

        if include_deltas:
            delta_mfcc = self.delta_transform(mfcc)
            feats["delta_mfcc"] = delta_mfcc.squeeze(0)

        if include_delta_deltas:
            delta2_mfcc = self.delta_transform(delta_mfcc)
            feats["delta2_mfcc"] = delta2_mfcc.squeeze(0)

        # Apply dataset train statistics if provided
        if norm_stats is not None:
            for k in feats:
                if k in norm_stats:
                    mean, std = norm_stats[k]
                    feats[k] = (feats[k] - mean) / (std + 1e-7)

        return feats
