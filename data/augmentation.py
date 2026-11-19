"""
Audio augmentation module for train-time data augmentation.
Includes additive noise, gain variation, codec simulation (bandpass/quantization), and RawBoost.
Each component is individually toggleable for ablation studies.
"""

import copy
import random
import numpy as np
import torch
from scipy import signal


class AudioAugmenter:
    def __init__(self,
                 enable_noise: bool = True,
                 enable_gain: bool = True,
                 enable_codec: bool = True,
                 enable_rawboost: bool = False,
                 snr_db_range: tuple = (15.0, 35.0),
                 gain_db_range: tuple = (-6.0, 6.0),
                 sample_rate: int = 16000):
        self.enable_noise = enable_noise
        self.enable_gain = enable_gain
        self.enable_codec = enable_codec
        self.enable_rawboost = enable_rawboost
        self.snr_db_range = snr_db_range
        self.gain_db_range = gain_db_range
        self.sample_rate = sample_rate

    def apply_gain(self, wav: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(self.gain_db_range[0], self.gain_db_range[1])
        gain_lin = 10.0 ** (gain_db / 20.0)
        return wav * gain_lin

    def apply_additive_noise(self, wav: np.ndarray) -> np.ndarray:
        snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        signal_power = np.mean(wav ** 2) + 1e-12
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), wav.shape)
        return wav + noise

    def apply_codec_simulation(self, wav: np.ndarray) -> np.ndarray:
        """
        Simulate telephony/lossy codec via bandpass filtering (300 Hz - 3400 Hz or 7000 Hz)
        and mu-law / 8-bit quantization.
        """
        fs = self.sample_rate
        # Randomly choose narrowband (telephony ~3.4kHz) or wideband (~7kHz)
        highcut = np.random.choice([3400.0, 6800.0])
        lowcut = 300.0
        
        # Butterworth bandpass filter
        nyq = 0.5 * fs
        b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype="band")
        filtered = signal.lfilter(b, a, wav)

        # 8-bit quantization noise simulation
        levels = 256.0
        quantized = np.round(filtered * (levels / 2.0)) / (levels / 2.0)
        return quantized

    def apply_rawboost(self, wav: np.ndarray) -> np.ndarray:
        """
        Impulsive signal dependent noise from RawBoost (Tak et al. ICASSP 2022).
        """
        p = np.random.uniform(0.1, 0.5)
        wav_copy = copy.deepcopy(wav)
        n_samples = int(len(wav) * (p / 100.0))
        if n_samples > 0:
            indices = np.random.choice(len(wav), size=n_samples, replace=False)
            rand_factor = (2.0 * np.random.rand(n_samples) - 1.0) * (2.0 * np.random.rand(n_samples) - 1.0)
            wav_copy[indices] = wav_copy[indices] + rand_factor * np.std(wav)
        return wav_copy

    def augment(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Apply enabled augmentations to 1D waveform torch.Tensor.
        """
        x = wav.cpu().numpy().astype(np.float32)

        if self.enable_gain and random.random() < 0.5:
            x = self.apply_gain(x)

        if self.enable_noise and random.random() < 0.5:
            x = self.apply_additive_noise(x)

        if self.enable_codec and random.random() < 0.3:
            x = self.apply_codec_simulation(x)

        if self.enable_rawboost and random.random() < 0.3:
            x = self.apply_rawboost(x)

        # Peak normalization after augmentation
        max_val = np.max(np.abs(x))
        if max_val > 1e-7:
            x = x / max_val

        return torch.from_numpy(x).to(wav.dtype)
