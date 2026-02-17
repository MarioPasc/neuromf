"""Explore FOMO-60K filter combinations and their effect on volume count.

Prints a summary table showing how many volumes each combination of
group filters and primary_scan_only produces, broken down by dataset.

Usage:
    ~/.conda/envs/neuromf/bin/python scripts/explore_fomo60k_filters.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

FOMO_ROOT = Path("/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K")


def count_volumes(
    participants: pd.DataFrame,
    mapping: pd.DataFrame,
    dataset_names: list[str],
    group_filter: list[str] | None,
    sequences: list[str],
    primary_scan_only: bool,
) -> dict[str, int]:
    """Count volumes per dataset for a given filter configuration.

    Args:
        participants: Full participants DataFrame.
        mapping: Full mapping DataFrame.
        dataset_names: Dataset names to include.
        group_filter: Group labels to include, or None for no filtering.
        sequences: Sequence types to match.
        primary_scan_only: If True, only exact ``{seq}.nii.gz``.

    Returns:
        Dict of dataset_name -> volume count.
    """
    # Filter participants by dataset
    part = participants[participants["dataset"].isin(dataset_names)].copy()

    # Apply group filter if specified
    if group_filter is not None:
        allowed = set(group_filter)
        keep = []
        for _, row in part.iterrows():
            group_val = row.get("group", "")
            if pd.isna(group_val) or str(group_val).strip() == "":
                group_val = ""
            else:
                group_val = str(group_val).strip()
            keep.append(group_val in allowed)
        part = part[keep]

    # Filter mapping by dataset and sequence
    mp = mapping[mapping["dataset"].isin(dataset_names)].copy()

    seq_patterns = []
    for seq in sequences:
        if primary_scan_only:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}\.nii\.gz$"))
        else:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}(_\d+)?\.nii\.gz$"))

    seq_mask = mp["filename"].apply(lambda fn: any(p.match(fn) for p in seq_patterns))
    mp = mp[seq_mask]

    # Inner join
    join_keys = ["dataset", "participant_id", "session_id"]
    merged = mp.merge(part[join_keys], on=join_keys, how="inner")

    # Count existing files per dataset
    counts: dict[str, int] = {ds: 0 for ds in dataset_names}
    for _, row in merged.iterrows():
        path = (
            FOMO_ROOT / row["dataset"] / row["participant_id"] / row["session_id"] / row["filename"]
        )
        if path.exists():
            counts[row["dataset"]] += 1

    return counts


def count_volumes_and_patients(
    participants: pd.DataFrame,
    mapping: pd.DataFrame,
    dataset_names: list[str],
    group_filter: list[str] | None,
    sequences: list[str],
    primary_scan_only: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    """Count volumes and unique patients per dataset for a given filter config.

    Args:
        participants: Full participants DataFrame.
        mapping: Full mapping DataFrame.
        dataset_names: Dataset names to include.
        group_filter: Group labels to include, or None for no filtering.
        sequences: Sequence types to match.
        primary_scan_only: If True, only exact ``{seq}.nii.gz``.

    Returns:
        Tuple of (scan_counts, patient_counts), each dict of dataset_name -> count.
    """
    # Filter participants by dataset
    part = participants[participants["dataset"].isin(dataset_names)].copy()

    # Apply group filter if specified
    if group_filter is not None:
        allowed = set(group_filter)
        keep = []
        for _, row in part.iterrows():
            group_val = row.get("group", "")
            if pd.isna(group_val) or str(group_val).strip() == "":
                group_val = ""
            else:
                group_val = str(group_val).strip()
            keep.append(group_val in allowed)
        part = part[keep]

    # Filter mapping by dataset and sequence
    mp = mapping[mapping["dataset"].isin(dataset_names)].copy()

    seq_patterns = []
    for seq in sequences:
        if primary_scan_only:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}\.nii\.gz$"))
        else:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}(_\d+)?\.nii\.gz$"))

    seq_mask = mp["filename"].apply(lambda fn: any(p.match(fn) for p in seq_patterns))
    mp = mp[seq_mask]

    # Inner join
    join_keys = ["dataset", "participant_id", "session_id"]
    merged = mp.merge(part[join_keys], on=join_keys, how="inner")

    # Count existing files and unique patients per dataset
    scan_counts: dict[str, int] = {ds: 0 for ds in dataset_names}
    patient_sets: dict[str, set[str]] = {ds: set() for ds in dataset_names}
    for _, row in merged.iterrows():
        path = (
            FOMO_ROOT / row["dataset"] / row["participant_id"] / row["session_id"] / row["filename"]
        )
        if path.exists():
            scan_counts[row["dataset"]] += 1
            patient_sets[row["dataset"]].add(row["participant_id"])

    patient_counts = {ds: len(pset) for ds, pset in patient_sets.items()}
    return scan_counts, patient_counts


def main() -> None:
    """Run filter exploration."""
    participants = pd.read_csv(FOMO_ROOT / "participants.tsv", sep="\t")
    mapping = pd.read_csv(FOMO_ROOT / "mapping.tsv", sep="\t")

    all_datasets = sorted(participants["dataset"].unique())

    # Show available group labels per dataset
    print("=" * 80)
    print("AVAILABLE GROUP LABELS PER DATASET")
    print("=" * 80)
    for ds in all_datasets:
        subset = participants[participants["dataset"] == ds]
        groups = subset["group"].fillna("<empty>").value_counts()
        n_part = len(subset)
        print(f"\n{ds} ({n_part} participants):")
        for grp, cnt in groups.items():
            print(f"  {grp:30s} {cnt:5d}")

    # Show available sequences per dataset
    print("\n" + "=" * 80)
    print("AVAILABLE SEQUENCES PER DATASET")
    print("=" * 80)
    for ds in all_datasets:
        subset = mapping[mapping["dataset"] == ds]
        seqs = subset["filename"].apply(lambda f: f.replace(".nii.gz", "")).value_counts()
        print(f"\n{ds} ({len(subset)} mapping entries):")
        for seq, cnt in seqs.head(10).items():
            print(f"  {seq:30s} {cnt:5d}")
        if len(seqs) > 10:
            print(f"  ... and {len(seqs) - 10} more unique filenames")

    # Define filter scenarios to test
    scenarios = [
        {
            "name": "Current config (Control/Nondemented/empty, primary_scan_only=false)",
            "groups": ["Control", "Nondemented", ""],
            "primary_scan_only": False,
        },
        {
            "name": "Add 'Brain Tumor' (primary_scan_only=false)",
            "groups": ["Control", "Nondemented", "", "Brain Tumor"],
            "primary_scan_only": False,
        },
        {
            "name": "No group filter (all groups, primary_scan_only=false)",
            "groups": None,
            "primary_scan_only": False,
        },
        {
            "name": "No group filter, primary_scan_only=true",
            "groups": None,
            "primary_scan_only": True,
        },
        {
            "name": "No group filter, primary_scan_only=false",
            "groups": None,
            "primary_scan_only": False,
        },
    ]

    print("\n" + "=" * 80)
    print("FILTER SCENARIOS — VOLUME COUNTS")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        counts = count_volumes(
            participants,
            mapping,
            all_datasets,
            group_filter=scenario["groups"],
            sequences=["t1"],
            primary_scan_only=scenario["primary_scan_only"],
        )
        total = 0
        for ds in all_datasets:
            n = counts.get(ds, 0)
            total += n
            print(f"  {ds:25s} {n:6d}")
        print(f"  {'TOTAL':25s} {total:6d}")

    # Also show what happens if we include all sequences
    print("\n--- All groups, all sequences (t1+t2+flair+...), primary_scan_only=false ---")
    # Get all unique sequence prefixes
    all_seqs = sorted(
        mapping["filename"].apply(lambda f: re.sub(r"(_\d+)?\.nii\.gz$", "", f)).unique()
    )
    print(f"  Available sequences: {all_seqs}")
    counts = count_volumes(
        participants,
        mapping,
        all_datasets,
        group_filter=None,
        sequences=all_seqs,
        primary_scan_only=False,
    )
    total = sum(counts.values())
    for ds in all_datasets:
        n = counts.get(ds, 0)
        print(f"  {ds:25s} {n:6d}")
    print(f"  {'TOTAL':25s} {total:6d}")

    # ---- Per-dataset breakdown from configs/fomo60k.yaml ----
    config_path = Path(__file__).resolve().parent.parent / "configs" / "fomo60k.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fomo_cfg = cfg["fomo60k"]
    cfg_sequences = fomo_cfg.get("sequences", ["t1"])
    cfg_primary = fomo_cfg.get("primary_scan_only", False)
    cfg_datasets = fomo_cfg["datasets"]  # list of {name, groups}

    print("\n" + "=" * 80)
    print("FOMO60K.YAML CONFIG — PER-DATASET UNIQUE PATIENTS & SCANS")
    print("=" * 80)
    print(f"  sequences={cfg_sequences}  primary_scan_only={cfg_primary}")
    print(f"  {'Dataset':25s} {'Patients':>10s} {'Scans':>10s}  Groups")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}  {'-' * 30}")

    total_patients = 0
    total_scans = 0
    for ds_entry in cfg_datasets:
        ds_name = ds_entry["name"]
        ds_groups = ds_entry.get("groups")
        scan_counts, patient_counts = count_volumes_and_patients(
            participants,
            mapping,
            [ds_name],
            group_filter=ds_groups,
            sequences=cfg_sequences,
            primary_scan_only=cfg_primary,
        )
        n_scans = scan_counts.get(ds_name, 0)
        n_patients = patient_counts.get(ds_name, 0)
        total_scans += n_scans
        total_patients += n_patients
        grp_str = str(ds_groups) if ds_groups is not None else "<all>"
        print(f"  {ds_name:25s} {n_patients:10d} {n_scans:10d}  {grp_str}")

    print(f"  {'-' * 25} {'-' * 10} {'-' * 10}")
    print(f"  {'TOTAL':25s} {total_patients:10d} {total_scans:10d}")


if __name__ == "__main__":
    main()
