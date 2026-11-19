"""
Manifest builder for EmoFake and ASVspoof datasets.
Outputs CSV with columns: audio_path, spoof_label, emotion_label, speaker, system, split.
Guarantees strict unseen-system isolation (S3-S7 reserved for test).

Supports two data layouts:
  A) Real EmoFake archive: English/{train,dev,test}/ dirs with label.txt files
  B) Flat or smoke directory: all wav files under a single tree
"""

import os
import re
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


VALID_EMOTIONS = {"neutral", "happy", "angry", "sad", "surprise"}


def parse_emofake_filename(filename: str):
    """
    Parse EmoFake filename into (is_spoof, emotion, speaker, system).
    Patterns:
      Fake: S1_Neutral2Happy_0011_000001.wav or S3_Neutral2Sad_0012_000005.wav
      Real: 0011_Happy_000001.wav or 0011_000001.wav
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if parts[0].startswith("S") and len(parts) >= 3:
        # Fake file
        sys_id = parts[0]
        # parts[1] is e.g. Neutral2Happy
        if "2" in parts[1]:
            tgt_emo = parts[1].split("2")[1].lower()
        else:
            tgt_emo = parts[1].lower()
        spk = parts[2]
        return True, tgt_emo, spk, sys_id
    elif len(parts) >= 2:
        # Real file: 0011_Happy_000001.wav
        spk = parts[0]
        # Check if second part is an emotion name
        candidate_emo = parts[1].lower()
        if candidate_emo in VALID_EMOTIONS:
            return False, candidate_emo, spk, "bonafide"
        else:
            # Might be a file without emotion in name, default to unknown
            return False, "unknown", spk, "bonafide"
    else:
        return None


def build_manifest_from_emofake_splits(data_dir: Path, output_csv: Path, language: str = "English"):
    """
    Build manifest from the official EmoFake archive structure.
    Expected layout: data_dir/{language}/{train,dev,test}/ with label.txt and wav files.
    
    label.txt format: filename.wav genuine/fake  (one per line)
    
    Protocol:
      - train/dev directories use S1, S2 (and bonafide)
      - test directory uses S3-S7 (and bonafide)
    """
    data_dir = Path(data_dir).resolve()
    records = []
    
    # Try to find the language directory
    lang_dir = data_dir / language
    if not lang_dir.exists():
        # Maybe the directory itself IS the language dir or root
        # Try common structures
        for candidate in [data_dir, data_dir / "emofake", data_dir / "EmoFake"]:
            if (candidate / "train").exists() or (candidate / language).exists():
                if (candidate / language).exists():
                    lang_dir = candidate / language
                else:
                    lang_dir = candidate
                break
    
    if not lang_dir.exists():
        print(f"WARNING: Could not find {language} directory. Looking for split dirs directly...")
        lang_dir = data_dir
    
    for split_name in ["train", "dev", "test"]:
        split_dir = lang_dir / split_name
        if not split_dir.exists():
            print(f"  Split directory not found: {split_dir}")
            continue
        
        # Read label.txt if it exists
        label_map = {}
        label_file = split_dir / "label.txt"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        fname = parts[0]
                        label = parts[1].lower()
                        label_map[fname] = label
                        label_map[fname.replace(".wav", "")] = label
            print(f"  Read {len(label_map)//2} labels from {label_file}")
        
        # Scan for wav files
        wav_files = sorted(list(split_dir.rglob("*.wav")))
        print(f"  Found {len(wav_files)} wav files in {split_dir}")
        
        for fpath in wav_files:
            fname = fpath.name
            fname_stem = fpath.stem
            
            # Try parsing filename
            parsed = parse_emofake_filename(fname)
            
            if parsed is not None:
                is_spoof, emotion, speaker, sys_id = parsed
            else:
                # Fallback: use label map
                is_spoof = True  # default
                emotion = "unknown"
                speaker = "unknown"
                sys_id = "unknown"
            
            # Override with label.txt if available
            if label_map:
                label = label_map.get(fname, label_map.get(fname_stem, None))
                if label is not None:
                    is_spoof = (label not in ["genuine", "bonafide", "real"])
            
            spoof_lbl = "spoof" if is_spoof else "bona_fide"
            
            records.append({
                "audio_path": str(fpath),
                "spoof_label": spoof_lbl,
                "emotion_label": emotion if emotion in VALID_EMOTIONS else "unknown",
                "speaker": speaker,
                "system": sys_id,
                "split": split_name
            })
    
    if not records:
        raise ValueError(f"No audio files found in {data_dir}. Check directory structure.")
    
    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nBuilt manifest with {len(df)} records at {output_csv}")
    print("\nSplit Distribution:")
    print(df.groupby(["split", "spoof_label"]).size().unstack(fill_value=0))
    print("\nEmotion Distribution:")
    print(df.groupby(["split", "emotion_label"]).size().unstack(fill_value=0))
    print("\nSystem Distribution:")
    print(df.groupby(["system", "split"]).size().unstack(fill_value=0))
    return df


def build_manifest_from_directory(data_dir: Path, output_csv: Path, dev_ratio: float = 0.2, seed: int = 42):
    """Build manifest from flat directory (e.g. synthetic smoke dataset)."""
    data_dir = Path(data_dir).resolve()
    audio_files = sorted(list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.flac")))
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")

    records = []
    rng = np.random.default_rng(seed)

    for fpath in audio_files:
        parsed = parse_emofake_filename(fpath.name)
        if parsed is None:
            # Fallback for ASVspoof or other files
            stem = fpath.stem.lower()
            is_spoof = ("spoof" in stem or "fake" in stem)
            spoof_lbl = "spoof" if is_spoof else "bona_fide"
            records.append({
                "audio_path": str(fpath),
                "spoof_label": spoof_lbl,
                "emotion_label": "neutral",
                "speaker": "unknown",
                "system": "asvspoof",
                "split": "train"
            })
            continue

        is_spoof, emotion, speaker, sys_id = parsed
        spoof_lbl = "spoof" if is_spoof else "bona_fide"

        # Protocol Split Logic:
        # Unseen systems S3, S4, S5, S6, S7 MUST be strictly in test split!
        if sys_id in ["S3", "S4", "S5", "S6", "S7"]:
            split = "test"
        elif sys_id in ["S1", "S2"]:
            # Train / dev partition for S1, S2
            split = "dev" if rng.random() < dev_ratio else "train"
        else:
            # Bona fide / genuine audio
            # Split proportionally so test has genuine audio for S3-S7 evaluation
            rand_val = rng.random()
            if rand_val < dev_ratio:
                split = "dev"
            elif rand_val < dev_ratio + 0.35:
                split = "test"
            else:
                split = "train"

        records.append({
            "audio_path": str(fpath),
            "spoof_label": spoof_lbl,
            "emotion_label": emotion if emotion in VALID_EMOTIONS else "unknown",
            "speaker": speaker,
            "system": sys_id,
            "split": split
        })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Built manifest with {len(df)} records at {output_csv}")
    print("\nSplit Distribution:")
    print(df.groupby(["split", "spoof_label"]).size().unstack(fill_value=0))
    print("\nSystem Distribution:")
    print(df.groupby(["system", "split"]).size().unstack(fill_value=0))
    return df


def validate_manifest(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = ["audio_path", "spoof_label", "emotion_label", "speaker", "system", "split"]
    for col in required:
        if col not in df.columns:
            raise AssertionError(f"Missing required column: {col}")

    # Check unseen system leakage
    train_dev_systems = set(df[df["split"].isin(["train", "dev"])]["system"].unique())
    unseen_systems = {"S3", "S4", "S5", "S6", "S7"}
    leaked = train_dev_systems.intersection(unseen_systems)
    if leaked:
        raise AssertionError(f"CRITICAL LEAKAGE DETECTED: Unseen test systems {leaked} found in train/dev!")

    # Check test has bonafide samples
    test_df = df[df["split"] == "test"]
    if len(test_df) > 0:
        n_bonafide_test = len(test_df[test_df["spoof_label"] == "bona_fide"])
        n_spoof_test = len(test_df[test_df["spoof_label"] == "spoof"])
        print(f"Test split: {n_bonafide_test} bonafide, {n_spoof_test} spoof")
        if n_bonafide_test == 0:
            print("WARNING: No bonafide samples in test split - EER cannot be computed!")
        if n_spoof_test == 0:
            print("WARNING: No spoof samples in test split - EER cannot be computed!")
    
    # Check emotion distribution
    emo_counts = df["emotion_label"].value_counts()
    print(f"\nEmotion distribution: {dict(emo_counts)}")

    print(f"\nValidation PASSED: {len(df)} rows. Zero unseen system leakage.")


def main():
    parser = argparse.ArgumentParser(description="Build and validate manifest CSV")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output", type=str, default="data/manifest.csv", help="Output manifest CSV path")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Proportion of S1/S2 data for dev")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing CSV")
    parser.add_argument("--use-splits", action="store_true", 
                        help="Use official train/dev/test split directories (real EmoFake archive)")
    parser.add_argument("--language", type=str, default="English", help="Language subset to use")
    args = parser.parse_args()

    out_p = Path(args.output)
    if args.validate_only:
        validate_manifest(out_p)
    elif args.use_splits:
        build_manifest_from_emofake_splits(Path(args.data_dir), out_p, language=args.language)
        validate_manifest(out_p)
    else:
        build_manifest_from_directory(Path(args.data_dir), out_p, dev_ratio=args.dev_ratio)
        validate_manifest(out_p)


if __name__ == "__main__":
    main()
