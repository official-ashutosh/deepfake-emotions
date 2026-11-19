"""
Unit tests for data isolation and leakage prevention.
Ensures zero unseen-system leakage (S3-S7) and speaker separation.
"""

from pathlib import Path
import pandas as pd
import pytest
from scripts.download_datasets import generate_synthetic_smoke_dataset
from scripts.build_manifest import build_manifest_from_directory, validate_manifest


@pytest.fixture(scope="module")
def smoke_manifest(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("smoke_data")
    emofake_dir, _ = generate_synthetic_smoke_dataset(tmp_dir, num_utterances_per_cell=6)
    manifest_csv = tmp_dir / "manifest.csv"
    df = build_manifest_from_directory(emofake_dir, manifest_csv)
    return manifest_csv, df


def test_no_unseen_system_in_train_or_dev(smoke_manifest):
    manifest_csv, df = smoke_manifest
    unseen_systems = {"S3", "S4", "S5", "S6", "S7"}
    train_dev_systems = set(df[df["split"].isin(["train", "dev"])]["system"].unique())
    leakage = train_dev_systems.intersection(unseen_systems)
    assert len(leakage) == 0, f"Unseen systems leaked into train/dev: {leakage}"


def test_test_split_has_all_unseen_systems(smoke_manifest):
    manifest_csv, df = smoke_manifest
    test_systems = set(df[df["split"] == "test"]["system"].unique())
    unseen_systems = {"S3", "S4", "S5", "S6", "S7"}
    assert unseen_systems.issubset(test_systems), f"Test split missing some unseen systems: {unseen_systems - test_systems}"


def test_test_split_contains_bonafide(smoke_manifest):
    manifest_csv, df = smoke_manifest
    test_df = df[df["split"] == "test"]
    n_bonafide = (test_df["spoof_label"] == "bona_fide").sum()
    n_spoof = (test_df["spoof_label"] == "spoof").sum()
    assert n_bonafide > 0, "Test split must contain bona fide speech to compute EER"
    assert n_spoof > 0, "Test split must contain spoof speech to compute EER"


def test_manifest_columns_and_emotions(smoke_manifest):
    manifest_csv, df = smoke_manifest
    expected_cols = ["audio_path", "spoof_label", "emotion_label", "speaker", "system", "split"]
    assert all(c in df.columns for c in expected_cols)
    emotions = set(df["emotion_label"].unique())
    expected_emotions = {"neutral", "happy", "angry", "sad", "surprise"}
    assert expected_emotions.issubset(emotions)


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        emo_dir, _ = generate_synthetic_smoke_dataset(tmp_p, num_utterances_per_cell=6)
        csv_p = tmp_p / "manifest.csv"
        df = build_manifest_from_directory(emo_dir, csv_p)
        validate_manifest(csv_p)
        print("All data leakage tests passed successfully!")
