"""
Dataset download and verification script for EmoFake and ASVspoof 2021.
Provides automated verification, download instructions/mirrors, and synthetic smoke dataset generation.
"""

import os
import sys
import argparse
import hashlib
import json
import urllib.request
import tarfile
import zipfile
import numpy as np
import soundfile as sf
from pathlib import Path


ZENODO_EMOFAKE_RECORD = "https://zenodo.org/records/12806228"
ZENODO_EMOFAKE_TAR = "https://zenodo.org/records/12806228/files/emofake.tar.gz"
GDRIVE_EMOFAKE_ID = "1aYPNVCVIBs6c9er_bhT3U8YzClF-EDsR"
ASVSPOOF_2021_URL = "https://www.asvspoof.org/index2021.html"


def compute_sha256(filepath: Path, block_size: int = 65536) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_synthetic_smoke_dataset(output_dir: Path, num_utterances_per_cell: int = 10, sample_rate: int = 16000):
    """
    Generate a structurally authentic mini-corpus mimicking EmoFake and ASVspoof.
    Includes:
    - 5 emotions: neutral, happy, angry, sad, surprise
    - Speakers: 0011, 0012, 0013, 0014 (English)
    - Systems: S1, S2 (train/dev) and S3, S4, S5, S6, S7 (unseen test)
    - Bona fide real files from ESD structure
    - Audio: 16 kHz mono wav files with synthetic harmonic formant carriers
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    emofake_dir = output_dir / "emofake_sample"
    emofake_dir.mkdir(parents=True, exist_ok=True)

    emotions = ["Neutral", "Happy", "Angry", "Sad", "Surprise"]
    train_speakers = ["0011", "0012", "0013"]
    test_speakers = ["0014", "0015"]
    all_speakers = train_speakers + test_speakers

    # Emotion pitch centers (Hz)
    emo_pitches = {
        "Neutral": 120.0,
        "Happy": 200.0,
        "Angry": 180.0,
        "Sad": 95.0,
        "Surprise": 240.0
    }

    print(f"Generating synthetic smoke dataset in: {emofake_dir}")
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)

    rng = np.random.default_rng(42)

    # 1. Generate Bona fide files: 0011_Happy_000001.wav
    real_dir = emofake_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    count_real = 0
    for spk in all_speakers:
        for emo in emotions:
            f0 = emo_pitches[emo] + (int(spk) % 10) * 15.0
            for idx in range(1, num_utterances_per_cell + 1):
                # Harmonic tone + slight envelope
                wave = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
                wave += 0.05 * rng.normal(0, 1, len(t))
                # Peak normalize
                wave = wave / np.max(np.abs(wave) + 1e-8)
                fname = f"{spk}_{emo}_{idx:06d}.wav"
                sf.write(str(real_dir / fname), wave.astype(np.float32), sample_rate)
                count_real += 1

    # 2. Generate Fake files for systems S1..S7: S1_Neutral2Happy_0011_000001.wav
    fake_dir = emofake_dir / "fake"
    fake_dir.mkdir(parents=True, exist_ok=True)
    count_fake = 0
    all_systems = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    for sys_id in all_systems:
        for spk in all_speakers:
            for src_emo in ["Neutral"]:
                for tgt_emo in emotions:
                    f0 = emo_pitches[tgt_emo] + (int(spk) % 10) * 15.0
                    for idx in range(1, max(2, num_utterances_per_cell // 2) + 1):
                        # Synthetic spoof artifact: robotic phase modulation / slight distortion
                        carrier = np.sin(2 * np.pi * f0 * t + 0.3 * np.sin(2 * np.pi * 50 * t))
                        wave = 0.5 * carrier + 0.1 * rng.normal(0, 1, len(t))
                        wave = np.clip(wave * 1.2, -0.95, 0.95)
                        wave = wave / np.max(np.abs(wave) + 1e-8)
                        fname = f"{sys_id}_{src_emo}2{tgt_emo}_{spk}_{idx:06d}.wav"
                        sf.write(str(fake_dir / fname), wave.astype(np.float32), sample_rate)
                        count_fake += 1

    # 3. Generate ASVspoof 2021 neutral mock directory
    asv_dir = output_dir / "asvspoof2021_sample"
    asv_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        # 10 bonafide, 10 spoof
        is_bona = (i % 2 == 0)
        f0 = 130.0 + i * 5.0
        wave = np.sin(2 * np.pi * f0 * t)
        if not is_bona:
            wave = wave + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        wave = wave / np.max(np.abs(wave) + 1e-8)
        lbl = "bonafide" if is_bona else "spoof"
        sf.write(str(asv_dir / f"LA_00{i:02d}_{lbl}.wav"), wave.astype(np.float32), sample_rate)

    print(f"Generated {count_real} bona fide and {count_fake} fake audio files in {emofake_dir}")
    print(f"Generated 20 sample files in {asv_dir}")
    return emofake_dir, asv_dir


def print_download_instructions():
    print("=" * 70)
    print("DATASET ACQUISITION & VERIFICATION INSTRUCTIONS")
    print("=" * 70)
    print("1. EmoFake Dataset (arXiv:2211.05363):")
    print(f"   - Official Zenodo record: {ZENODO_EMOFAKE_RECORD}")
    print(f"   - Zenodo direct tar download: {ZENODO_EMOFAKE_TAR}")
    print(f"   - Official Google Drive mirror: https://drive.google.com/file/d/{GDRIVE_EMOFAKE_ID}/view")
    print("   - Size: ~8.8 GB compressed archive containing real ESD speech and S1-S7 EVC fakes.")
    print("\n2. ASVspoof 2021 Benchmark (LA, DF):")
    print(f"   - Challenge portal: {ASVSPOOF_2021_URL}")
    print("   - Evaluation keys & protocols are included in `external/SSL_Anti-spoofing/`")
    print("\n3. Automated Smoke Testing:")
    print("   - Run: python scripts/download_datasets.py --synthetic-smoke --output data/smoke_dataset")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download and verify EmoFake and ASVspoof 2021 datasets")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Generate synthetic smoke dataset for fast testing and CI")
    parser.add_argument("--output", type=str, default="data/smoke_dataset", help="Output directory for smoke dataset or downloaded archives")
    parser.add_argument("--verify", type=str, default=None, help="Path to archive or directory to verify")
    args = parser.parse_args()

    if args.synthetic_smoke:
        generate_synthetic_smoke_dataset(Path(args.output))
        return

    if args.verify:
        p = Path(args.verify)
        if not p.exists():
            print(f"Error: Path {p} does not exist.")
            sys.exit(1)
        if p.is_file():
            print(f"Verifying file {p}...")
            size_gb = p.stat().st_size / (1024 ** 3)
            print(f"Size: {size_gb:.2f} GB")
            sha = compute_sha256(p)
            print(f"SHA-256: {sha}")
        else:
            wavs = list(p.rglob("*.wav")) + list(p.rglob("*.flac"))
            print(f"Directory {p} contains {len(wavs)} audio files.")
        return

    print_download_instructions()


if __name__ == "__main__":
    main()
