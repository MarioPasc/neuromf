"""FOMO-60K dataset loader with metadata-based filtering.

Reads FOMO-60K metadata TSVs (participants.tsv, mapping.tsv), applies
dataset/sequence/group filters, and returns MONAI-compatible file lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class FOMO60KDatasetFilter:
    """Filter for one FOMO-60K sub-dataset.

    Attributes:
        name: Dataset identifier, e.g. ``"PT001_OASIS1"``.
        groups: Exact-match group labels to include. Use ``""`` to match
            subjects with NaN/empty group labels.
    """

    name: str
    groups: list[str] = field(default_factory=list)


@dataclass
class FOMO60KConfig:
    """Top-level FOMO-60K data loading configuration.

    Attributes:
        root: Absolute path to the FOMO-60K root directory.
        datasets: Per-dataset filter specifications.
        sequences: Sequence types to include, e.g. ``["t1"]``.
        primary_scan_only: If True, only match exact ``{seq}.nii.gz``
            filenames, skipping duplicates like ``t1_2.nii.gz``.
    """

    root: Path
    datasets: list[FOMO60KDatasetFilter]
    sequences: list[str] = field(default_factory=lambda: ["t1"])
    primary_scan_only: bool = True

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> FOMO60KConfig:
        """Build a FOMO60KConfig from a merged OmegaConf config.

        Expects ``cfg.paths.fomo60k_root`` and ``cfg.fomo60k`` sections.

        Args:
            cfg: Merged OmegaConf config (base + fomo60k).

        Returns:
            Populated FOMO60KConfig instance.
        """
        fomo_cfg = cfg.fomo60k
        datasets = []
        for ds in fomo_cfg.datasets:
            groups = list(ds.groups) if ds.get("groups") else []
            datasets.append(FOMO60KDatasetFilter(name=ds.name, groups=groups))

        return cls(
            root=Path(cfg.paths.fomo60k_root),
            datasets=datasets,
            sequences=list(fomo_cfg.sequences),
            primary_scan_only=fomo_cfg.primary_scan_only,
        )


def get_fomo60k_file_list(
    config: FOMO60KConfig,
    n_volumes: int | None = None,
) -> list[dict[str, str]]:
    """Load FOMO-60K metadata, apply filters, return MONAI-format dicts.

    Args:
        config: FOMO-60K dataset configuration with filters.
        n_volumes: If set, return only the first *n_volumes* (sorted).

    Returns:
        Sorted list of ``[{"image": "/abs/path/to/t1.nii.gz"}, ...]``.

    Raises:
        FileNotFoundError: If metadata TSVs are missing or no files match.
    """
    root = config.root

    # Read metadata
    participants_path = root / "participants.tsv"
    mapping_path = root / "mapping.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping.tsv not found at {mapping_path}")

    participants = pd.read_csv(participants_path, sep="\t")
    mapping = pd.read_csv(mapping_path, sep="\t")

    # Build dataset name -> allowed groups lookup
    dataset_names = {ds.name for ds in config.datasets}
    dataset_groups = {ds.name: set(ds.groups) for ds in config.datasets}

    # 1. Filter participants by dataset and group
    part_mask = participants["dataset"].isin(dataset_names)
    participants_filtered = participants[part_mask].copy()

    # Apply per-dataset group filtering
    keep_rows = []
    for _, row in participants_filtered.iterrows():
        ds_name = row["dataset"]
        allowed_groups = dataset_groups[ds_name]
        if not allowed_groups:
            # No group filter — keep all subjects from this dataset
            keep_rows.append(True)
            continue
        group_val = row.get("group", "")
        # Treat NaN/empty as ""
        if pd.isna(group_val) or str(group_val).strip() == "":
            group_val = ""
        else:
            group_val = str(group_val).strip()
        keep_rows.append(group_val in allowed_groups)

    participants_filtered = participants_filtered[keep_rows]
    logger.info(
        "Participants after group filtering: %d (from %d datasets)",
        len(participants_filtered),
        len(dataset_names),
    )

    # 2. Filter mapping by dataset and sequence
    map_mask = mapping["dataset"].isin(dataset_names)
    mapping_filtered = mapping[map_mask].copy()

    # Filter by sequence pattern
    seq_patterns = []
    for seq in config.sequences:
        if config.primary_scan_only:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}\.nii\.gz$"))
        else:
            seq_patterns.append(re.compile(rf"^{re.escape(seq)}(_\d+)?\.nii\.gz$"))

    def matches_sequence(filename: str) -> bool:
        return any(p.match(filename) for p in seq_patterns)

    seq_mask = mapping_filtered["filename"].apply(matches_sequence)
    mapping_filtered = mapping_filtered[seq_mask]
    logger.info("Mapping entries after sequence filtering: %d", len(mapping_filtered))

    # 3. Inner join on (dataset, participant_id, session_id)
    join_keys = ["dataset", "participant_id", "session_id"]
    merged = mapping_filtered.merge(
        participants_filtered[join_keys],
        on=join_keys,
        how="inner",
    )
    logger.info("Files after join: %d", len(merged))

    # 4. Build paths and verify existence
    file_list: list[dict[str, str]] = []
    missing_count = 0
    for _, row in merged.iterrows():
        path = root / row["dataset"] / row["participant_id"] / row["session_id"] / row["filename"]
        if path.exists():
            file_list.append({"image": str(path)})
        else:
            missing_count += 1
            if missing_count <= 5:
                logger.warning("Missing file: %s", path)

    if missing_count > 5:
        logger.warning("... and %d more missing files", missing_count - 5)
    if missing_count > 0:
        logger.warning("Total missing files: %d", missing_count)

    if not file_list:
        raise FileNotFoundError(f"No matching .nii.gz files found in {root} with the given filters")

    # Sort for reproducibility
    file_list.sort(key=lambda d: d["image"])
    logger.info("FOMO-60K file list: %d volumes", len(file_list))

    if n_volumes is not None:
        file_list = file_list[:n_volumes]
        logger.info("Using first %d volumes", n_volumes)

    return file_list
