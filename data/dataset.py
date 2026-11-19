"""
Dataset and DataLoader utilities for EmoFake and ASVspoof.
Includes balanced batch sampling across spoof x emotion cells.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from data.preprocessing import AudioPreprocessor
from data.augmentation import AudioAugmenter


EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprise": 4
}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}


class AudioSpoofEmotionDataset(Dataset):
    def __init__(self,
                 manifest_path: Union[str, Path],
                 split: str = "train",
                 preprocessor: Optional[AudioPreprocessor] = None,
                 augmenter: Optional[AudioAugmenter] = None,
                 norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
                 return_raw: bool = False,
                 feature_type: str = "combined"):
        """
        Args:
            manifest_path: Path to manifest CSV.
            split: 'train', 'dev', or 'test'.
            preprocessor: AudioPreprocessor instance.
            augmenter: AudioAugmenter instance (applied only if split == 'train').
            norm_stats: Dict of mean and std computed from train set.
            return_raw: If True, returns raw waveform (for RawNet2, AASIST, SSL).
            feature_type: 'combined', 'mel_only', 'mfcc_only', 'mfcc_deltas'.
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.augmenter = augmenter if split == "train" else None
        self.norm_stats = norm_stats
        self.return_raw = return_raw
        self.feature_type = feature_type

        # Read manifest
        df = pd.read_csv(self.manifest_path)
        if split != "all":
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

        # Prepare label tensors
        self.spoof_labels = []
        self.emotion_labels = []
        self.audio_paths = []
        self.systems = []
        self.speakers = []

        for _, row in self.df.iterrows():
            # Spoof label: 1 = bona_fide, 0 = spoof
            sl = str(row["spoof_label"]).lower().strip()
            is_bona = 1 if sl in ["bona_fide", "bonafide", "real", "1"] else 0
            self.spoof_labels.append(is_bona)

            # Emotion label: 0..4 or -1 if unknown
            el = str(row.get("emotion_label", "")).lower().strip()
            emo_id = EMOTION_MAP.get(el, -1)
            self.emotion_labels.append(emo_id)

            self.audio_paths.append(str(row["audio_path"]))
            self.systems.append(str(row.get("system", "-")))
            self.speakers.append(str(row.get("speaker", "-")))

        self.spoof_labels = np.array(self.spoof_labels, dtype=np.int64)
        self.emotion_labels = np.array(self.emotion_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        path = self.audio_paths[index]
        is_train = (self.split == "train")

        # Load raw audio
        wav = self.preprocessor.load_audio(path, is_train=is_train)

        # Apply augmentation if training
        if self.augmenter is not None:
            wav = self.augmenter.augment(wav)

        spoof_label = torch.tensor(self.spoof_labels[index], dtype=torch.long)
        emotion_label = torch.tensor(self.emotion_labels[index], dtype=torch.long)

        item = {
            "path": path,
            "spoof_label": spoof_label,
            "emotion_label": emotion_label,
            "system": self.systems[index],
            "speaker": self.speakers[index],
            "wav": wav
        }

        if not self.return_raw:
            # Extract features
            feats = self.preprocessor.extract_features(
                wav,
                include_deltas=True,
                include_delta_deltas=False,
                norm_stats=self.norm_stats
            )
            # Assemble feature map based on feature_type
            # log_mel is (80, T), mfcc is (40, T), delta_mfcc is (40, T)
            mel = feats["log_mel"]
            mfcc = feats["mfcc"]
            delta = feats["delta_mfcc"]

            if self.feature_type == "mel_only":
                # shape (1, 80, T)
                feature_tensor = mel.unsqueeze(0)
            elif self.feature_type == "mfcc_only":
                # shape (1, 40, T)
                feature_tensor = mfcc.unsqueeze(0)
            elif self.feature_type == "mfcc_deltas":
                # shape (2, 40, T)
                feature_tensor = torch.stack([mfcc, delta], dim=0)
            else:  # "combined"
                # Pad MFCC to 80 bins or stack along channel
                # Clean stack: (3 channels: mel 80xT, mfcc_padded 80xT, delta_padded 80xT)
                mfcc_pad = torch.nn.functional.pad(mfcc, (0, 0, 0, mel.shape[0] - mfcc.shape[0]))
                delta_pad = torch.nn.functional.pad(delta, (0, 0, 0, mel.shape[0] - delta.shape[0]))
                feature_tensor = torch.stack([mel, mfcc_pad, delta_pad], dim=0)

            item["features"] = feature_tensor

        return item


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler across (spoof_label x emotion_label) groups.
    Ensures each batch samples evenly across active cells.
    """
    def __init__(self, dataset: AudioSpoofEmotionDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by (spoof, emotion)
        self.groups = {}
        for idx in range(len(dataset)):
            sp = int(dataset.spoof_labels[idx])
            em = int(dataset.emotion_labels[idx])
            key = (sp, em)
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)

        self.keys = list(self.groups.keys())
        self.n_groups = len(self.keys)
        # Total batches per epoch
        self.total_samples = len(dataset)
        self.n_batches = self.total_samples // batch_size

    def __iter__(self):
        # Shuffle within each group
        shuffled = {k: np.random.permutation(v).tolist() for k, v in self.groups.items()}
        pointers = {k: 0 for k in self.groups}

        for _ in range(self.n_batches):
            batch = []
            while len(batch) < self.batch_size:
                for k in self.keys:
                    grp = shuffled[k]
                    ptr = pointers[k]
                    if ptr >= len(grp):
                        shuffled[k] = np.random.permutation(self.groups[k]).tolist()
                        grp = shuffled[k]
                        ptr = 0
                    batch.append(grp[ptr])
                    pointers[k] = ptr + 1
                    if len(batch) >= self.batch_size:
                        break
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def compute_dataset_norm_stats(dataset: AudioSpoofEmotionDataset, max_samples: int = 500) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and std over training partition features only.
    """
    n = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), size=n, replace=False)
    
    sums = {}
    sq_sums = {}
    counts = {}

    for idx in indices:
        wav = dataset.preprocessor.load_audio(dataset.audio_paths[idx], is_train=False)
        feats = dataset.preprocessor.extract_features(wav, include_deltas=True, include_delta_deltas=False)
        for k, v in feats.items():
            if k not in sums:
                sums[k] = 0.0
                sq_sums[k] = 0.0
                counts[k] = 0
            val = v.cpu().numpy()
            sums[k] += np.sum(val)
            sq_sums[k] += np.sum(val ** 2)
            counts[k] += val.size

    stats = {}
    for k in sums:
        mean = float(sums[k] / counts[k])
        var = float((sq_sums[k] / counts[k]) - (mean ** 2))
        std = float(np.sqrt(max(var, 1e-6)))
        stats[k] = (mean, std)

    return stats
