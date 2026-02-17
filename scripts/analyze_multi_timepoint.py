#!/usr/bin/env python3
"""Multi-timepoint analysis and data leakage simulation for FOMO-60K.

Reads mapping.tsv and participants.tsv, prints:
1. Per-dataset table (subjects, T1 scans, multi-T1 sessions, etc.)
2. Leakage simulation: file-level split shows leaked subjects
3. Fix verification: subject-level split shows 0 leakage

Usage:
    python scripts/analyze_multi_timepoint.py
    python scripts/analyze_multi_timepoint.py --fomo60k-root /path/to/FOMO60K
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd


def load_metadata(fomo60k_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load mapping.tsv and participants.tsv from FOMO-60K root.

    Args:
        fomo60k_root: Path to FOMO-60K dataset root.

    Returns:
        Tuple of (mapping, participants) DataFrames.
    """
    mapping_path = fomo60k_root / "mapping.tsv"
    participants_path = fomo60k_root / "participants.tsv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.tsv not found at {mapping_path}")
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")

    mapping = pd.read_csv(mapping_path, sep="\t")
    participants = pd.read_csv(participants_path, sep="\t")
    return mapping, participants


def analyze_dataset(mapping: pd.DataFrame, dataset: str) -> dict:
    """Compute multi-timepoint stats for one dataset.

    Args:
        mapping: Full mapping DataFrame.
        dataset: Dataset name, e.g. ``"PT001_OASIS1"``.

    Returns:
        Dict with analysis results.
    """
    dm = mapping[mapping["dataset"] == dataset].copy()
    # Filter T1 scans only
    t1_mask = dm["filename"].str.match(r"^t1(_\d+)?\.nii\.gz$")
    dm = dm[t1_mask]

    n_t1_scans = len(dm)
    unique_subjects = dm["participant_id"].nunique()
    unique_sessions = dm.groupby(["participant_id", "session_id"]).ngroups

    # Multi-T1 sessions: sessions with more than one T1 scan
    scans_per_session = dm.groupby(["participant_id", "session_id"]).size()
    multi_t1_sessions = int((scans_per_session > 1).sum())
    max_t1_per_session = int(scans_per_session.max()) if len(scans_per_session) > 0 else 0

    # Multi-session subjects: subjects with more than one session
    sessions_per_subject = dm.groupby("participant_id")["session_id"].nunique()
    multi_session_subjects = int((sessions_per_subject > 1).sum())

    return {
        "dataset": dataset,
        "subjects": unique_subjects,
        "sessions": unique_sessions,
        "t1_scans": n_t1_scans,
        "multi_t1_sessions": multi_t1_sessions,
        "max_t1_per_session": max_t1_per_session,
        "multi_session_subjects": multi_session_subjects,
    }


def simulate_file_level_split(
    entries: list[tuple[str, str]],
    split_ratio: float = 0.9,
    split_seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Simulate file-level split (the buggy approach).

    Args:
        entries: List of (subject_key, file_id) tuples.
        split_ratio: Fraction for training.
        split_seed: Random seed.

    Returns:
        Tuple of (train_subjects, val_subjects) sets.
    """
    indices = list(range(len(entries)))
    random.Random(split_seed).shuffle(indices)
    n_train = int(len(indices) * split_ratio)

    train_subjects = {entries[i][0] for i in indices[:n_train]}
    val_subjects = {entries[i][0] for i in indices[n_train:]}
    return train_subjects, val_subjects


def simulate_subject_level_split(
    entries: list[tuple[str, str]],
    split_ratio: float = 0.9,
    split_seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Simulate subject-level split (the correct approach).

    Args:
        entries: List of (subject_key, file_id) tuples.
        split_ratio: Fraction for training.
        split_seed: Random seed.

    Returns:
        Tuple of (train_subjects, val_subjects) sets.
    """
    subjects = sorted({e[0] for e in entries})
    random.Random(split_seed).shuffle(subjects)
    n_train = int(len(subjects) * split_ratio)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:])
    return train_subjects, val_subjects


def main() -> None:
    """Run multi-timepoint analysis and leakage simulation."""
    parser = argparse.ArgumentParser(description="FOMO-60K multi-timepoint analysis")
    parser.add_argument(
        "--fomo60k-root",
        type=Path,
        default=Path("/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K"),
        help="Path to FOMO-60K root directory",
    )
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Split seed")
    args = parser.parse_args()

    mapping, participants = load_metadata(args.fomo60k_root)

    # Target datasets (matching fomo60k.yaml)
    target_datasets = [
        "PT001_OASIS1",
        "PT002_OASIS2",
        "PT005_IXI",
        "PT007_NIMH",
        "PT008_DLBS",
        "PT011_MBSR",
        "PT012_UCLA",
        "PT015_NKI",
    ]

    # ── 1. Per-dataset table ──
    print("=" * 90)
    print("FOMO-60K Multi-Timepoint Analysis")
    print("=" * 90)
    print()
    header = f"{'Dataset':<16} {'Subjects':>8} {'Sessions':>9} {'T1 scans':>9} {'Multi-T1':>9} {'Max T1/ses':>11} {'Multi-ses':>10}"
    print(header)
    print("-" * len(header))

    all_results = []
    for ds in target_datasets:
        ds_mapping = mapping[mapping["dataset"] == ds]
        if ds_mapping.empty:
            print(f"{ds:<16} {'(not in mapping)':>8}")
            continue
        result = analyze_dataset(mapping, ds)
        all_results.append(result)
        print(
            f"{result['dataset']:<16} {result['subjects']:>8} {result['sessions']:>9} "
            f"{result['t1_scans']:>9} {result['multi_t1_sessions']:>9} "
            f"{result['max_t1_per_session']:>11} {result['multi_session_subjects']:>10}"
        )

    totals = {
        "subjects": sum(r["subjects"] for r in all_results),
        "t1_scans": sum(r["t1_scans"] for r in all_results),
    }
    print("-" * len(header))
    print(f"{'TOTAL':<16} {totals['subjects']:>8} {'':>9} {totals['t1_scans']:>9}")
    print()

    # ── 2. Build entries for leakage simulation ──
    # Each entry is (subject_key, file_identifier)
    entries: list[tuple[str, str]] = []
    for ds in target_datasets:
        dm = mapping[mapping["dataset"] == ds].copy()
        t1_mask = dm["filename"].str.match(r"^t1(_\d+)?\.nii\.gz$")
        dm = dm[t1_mask]
        for _, row in dm.iterrows():
            subject_key = f"{row['dataset']}_{row['participant_id']}"
            file_id = (
                f"{row['dataset']}/{row['participant_id']}/{row['session_id']}/{row['filename']}"
            )
            entries.append((subject_key, file_id))

    print(f"Total entries for splitting: {len(entries)}")
    print()

    # ── 3. File-level split (buggy) ──
    print("=" * 90)
    print(f"FILE-LEVEL SPLIT (seed={args.split_seed}, ratio={args.split_ratio})")
    print("=" * 90)
    train_subj, val_subj = simulate_file_level_split(entries, args.split_ratio, args.split_seed)
    leaked = train_subj & val_subj
    print(f"  Train subjects: {len(train_subj)}")
    print(f"  Val subjects:   {len(val_subj)}")
    print(f"  Leaked subjects (in BOTH): {len(leaked)}")
    if leaked:
        examples = sorted(leaked)[:10]
        print(f"  Examples: {', '.join(examples)}")
    print()

    # ── 4. Subject-level split (correct) ──
    print("=" * 90)
    print(f"SUBJECT-LEVEL SPLIT (seed={args.split_seed}, ratio={args.split_ratio})")
    print("=" * 90)
    train_subj_fix, val_subj_fix = simulate_subject_level_split(
        entries, args.split_ratio, args.split_seed
    )
    leaked_fix = train_subj_fix & val_subj_fix
    print(f"  Train subjects: {len(train_subj_fix)}")
    print(f"  Val subjects:   {len(val_subj_fix)}")
    print(f"  Leaked subjects (in BOTH): {len(leaked_fix)}")

    # Count files per split
    train_files = sum(1 for e in entries if e[0] in train_subj_fix)
    val_files = sum(1 for e in entries if e[0] in val_subj_fix)
    print(f"  Train files: {train_files}")
    print(f"  Val files:   {val_files}")
    print()

    if len(leaked_fix) > 0:
        print("ERROR: Subject-level split still has leakage!")
        sys.exit(1)
    else:
        print("OK: Subject-level split has ZERO leakage.")


if __name__ == "__main__":
    main()
