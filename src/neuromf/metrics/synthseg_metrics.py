"""SynthSeg-based morphological evaluation for generated brain MRI.

Wraps the SynthSeg predict script (or FreeSurfer ``mri_synthseg``) via
:mod:`subprocess` to avoid TensorFlow/Keras conflicts with the PyTorch
environment.  Converts HDF5 volumes to NIfTI files, runs SynthSeg, and
computes morphological metrics:

- **Success rate:** fraction of volumes segmented without error.
- **Regional volume correlation:** Pearson *r* between matched real/generated
  regional volumes (hippocampus, ventricles, cortex, etc.).
- **Regional volume KL:** KL divergence of per-region volume histograms.
- **SynthSeg Dice:** mean Dice overlap between paired label maps.

SynthSeg can be invoked in two ways (configured via ``synthseg.command``):

1. **Standalone repo** (recommended for HPC without FreeSurfer)::

       command:
         - "/path/to/synthseg_env/bin/python"
         - "/path/to/SynthSeg/scripts/commands/SynthSeg_predict.py"

2. **FreeSurfer binary**::

       command:
         - "mri_synthseg"

Both accept the same ``--i``, ``--o``, ``--vol``, ``--robust`` flags.

Reference:
    Billot et al., "SynthSeg: Segmentation of brain MRI scans of any contrast
    and resolution without retraining", Medical Image Analysis, 2023.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)

# FreeSurfer label IDs for regions of interest
SYNTHSEG_LABEL_MAP: dict[str, int] = {
    "Left-Hippocampus": 17,
    "Right-Hippocampus": 53,
    "Left-Lateral-Ventricle": 4,
    "Right-Lateral-Ventricle": 43,
    "Left-Cerebral-Cortex": 3,
    "Right-Cerebral-Cortex": 42,
    "Left-Thalamus": 10,
    "Left-Caudate": 11,
    "Left-Putamen": 12,
    "Left-Cerebellum-Cortex": 8,
}


@dataclass
class SynthSegConfig:
    """Configuration for SynthSeg runner.

    Args:
        command: Command prefix list for invoking SynthSeg.  Can be a single
            binary (``["mri_synthseg"]``) or a Python interpreter + script
            (``["/path/to/python", "/path/to/SynthSeg_predict.py"]``).
        regions_of_interest: FreeSurfer region names to evaluate.
        robust: Use robust mode for challenging scans.
        parc: Enable cortical parcellation.
        voxel_size_mm: Isotropic voxel size for the NIfTI affine.
        threads: Number of CPU threads for SynthSeg (0 = use default).
        crop: Crop size for SynthSeg (must be divisible by 32, 0 = default 192).
    """

    command: list[str] = field(default_factory=lambda: ["mri_synthseg"])
    regions_of_interest: list[str] = field(default_factory=lambda: list(SYNTHSEG_LABEL_MAP.keys()))
    robust: bool = True
    parc: bool = False
    voxel_size_mm: float = 1.0
    threads: int = 0
    crop: int = 0


def check_synthseg_available(config: SynthSegConfig) -> bool:
    """Check if the SynthSeg command is available.

    For a single binary (e.g. ``["mri_synthseg"]``), checks PATH.
    For a Python script (e.g. ``["/path/python", "/path/script.py"]``),
    checks that the interpreter exists and the script file exists.

    Args:
        config: SynthSeg configuration.

    Returns:
        True if the command can be executed.
    """
    cmd = config.command
    if not cmd:
        return False

    # Check the executable (first element)
    if not Path(cmd[0]).is_absolute():
        if shutil.which(cmd[0]) is None:
            return False
    elif not Path(cmd[0]).exists():
        return False

    # If there's a script path (second element), check it exists
    if len(cmd) > 1 and not Path(cmd[1]).exists():
        return False

    return True


def _h5_volumes_to_nifti(
    h5_path: Path,
    output_dir: Path,
    voxel_size_mm: float = 1.0,
    max_volumes: int | None = None,
) -> list[Path]:
    """Convert volumes from HDF5 archive to NIfTI files.

    Args:
        h5_path: Path to volume HDF5 with ``/volumes`` dataset.
        output_dir: Directory for output NIfTI files.
        voxel_size_mm: Isotropic voxel spacing for the NIfTI affine.
        max_volumes: Maximum number of volumes to convert.

    Returns:
        List of paths to created NIfTI files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([voxel_size_mm, voxel_size_mm, voxel_size_mm, 1.0])

    paths: list[Path] = []
    with h5py.File(str(h5_path), "r") as f:
        n = f["volumes"].shape[0]
        if max_volumes is not None:
            n = min(n, max_volumes)

        for i in range(n):
            nii_path = output_dir / f"vol_{i:04d}.nii.gz"
            if nii_path.exists():
                paths.append(nii_path)
                continue
            vol = f["volumes"][i].astype(np.float32)  # (D, H, W)
            nii = nib.Nifti1Image(vol, affine)
            nib.save(nii, str(nii_path))
            paths.append(nii_path)

    logger.info("Converted %d volumes from %s to NIfTI", len(paths), h5_path.name)
    return paths


def run_synthseg(
    input_dir: Path,
    output_dir: Path,
    volumes_csv: Path,
    config: SynthSegConfig,
    qc_csv: Path | None = None,
) -> bool:
    """Run SynthSeg on a directory of NIfTI volumes.

    Args:
        input_dir: Directory containing input ``.nii.gz`` files.
        output_dir: Directory for output segmentation label maps.
        volumes_csv: Path to output CSV with regional volumes.
        config: SynthSeg configuration.
        qc_csv: Optional path for QC scores CSV.

    Returns:
        True if SynthSeg completed successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *config.command,
        "--i",
        str(input_dir),
        "--o",
        str(output_dir),
        "--vol",
        str(volumes_csv),
    ]
    if config.robust:
        cmd.append("--robust")
    if config.parc:
        cmd.append("--parc")
    if config.threads > 0:
        cmd.extend(["--threads", str(config.threads)])
    if config.crop > 0:
        cmd.extend(["--crop", str(config.crop)])
    if qc_csv is not None:
        cmd.extend(["--qc", str(qc_csv)])

    logger.info("Running SynthSeg: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14400,  # 4-hour timeout for 2000 volumes
        )
        if result.returncode != 0:
            logger.error(
                "SynthSeg failed (rc=%d):\nstdout: %s\nstderr: %s",
                result.returncode,
                result.stdout[-500:] if result.stdout else "",
                result.stderr[-500:] if result.stderr else "",
            )
            return False
        logger.info("SynthSeg completed successfully")
        return True
    except FileNotFoundError:
        logger.error("SynthSeg command not found: %s", config.command)
        return False
    except subprocess.TimeoutExpired:
        logger.error("SynthSeg timed out after 4 hours")
        return False


def parse_volumes_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    """Parse SynthSeg volumes CSV into a nested dict.

    Args:
        csv_path: Path to the volumes CSV output by SynthSeg ``--vol``.

    Returns:
        Dict mapping ``{subject_name: {region_name: volume_mm3}}``.
    """
    volumes: dict[str, dict[str, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # SynthSeg CSV: first column is subject/filename, rest are region volumes
            subject = row.get("subject", row.get("", ""))
            region_vols: dict[str, float] = {}
            for key, val in row.items():
                if key in ("subject", ""):
                    continue
                try:
                    region_vols[key] = float(val)
                except (ValueError, TypeError):
                    pass
            volumes[subject] = region_vols

    return volumes


def validate_volumes_csv(
    volumes: dict[str, dict[str, float]],
) -> tuple[bool, str]:
    """Check if SynthSeg produced meaningful results (not all zeros).

    Args:
        volumes: Parsed CSV dict from ``parse_volumes_csv()``.

    Returns:
        Tuple of ``(is_valid, message)``.  ``is_valid`` is False if ALL
        subjects have ALL regions equal to 0.0.
    """
    if not volumes:
        return False, "Empty volumes CSV — no subjects found"

    all_zero = True
    for subject, regions in volumes.items():
        for region, val in regions.items():
            if abs(val) > 1e-10:
                all_zero = False
                break
        if not all_zero:
            break

    if all_zero:
        n_subjects = len(volumes)
        return False, (
            f"All {n_subjects} subjects have all-zero regional volumes — likely blank input data"
        )

    return True, "OK"


def compute_success_rate(
    input_dir: Path,
    output_dir: Path,
) -> float:
    """Compute fraction of volumes successfully segmented by SynthSeg.

    Args:
        input_dir: Directory of input NIfTI files.
        output_dir: Directory of SynthSeg output label maps.

    Returns:
        Success rate in [0, 1].
    """
    input_files = sorted(input_dir.glob("*.nii.gz"))
    if not input_files:
        return 0.0

    n_success = 0
    for nii_path in input_files:
        label_path = output_dir / nii_path.name
        if label_path.exists() and label_path.stat().st_size > 0:
            n_success += 1

    rate = n_success / len(input_files)
    logger.info(
        "SynthSeg success rate: %d / %d (%.2f%%)",
        n_success,
        len(input_files),
        rate * 100,
    )
    return rate


def compute_dice_from_labels(
    real_label_path: Path,
    gen_label_path: Path,
    label_ids: list[int] | None = None,
) -> dict[str, float]:
    """Compute Dice overlap between two SynthSeg label maps.

    Args:
        real_label_path: Path to real NIfTI label map.
        gen_label_path: Path to generated NIfTI label map.
        label_ids: Label IDs to compute Dice for. If None, uses all
            labels in ``SYNTHSEG_LABEL_MAP``.

    Returns:
        Dict with per-region Dice and ``"mean"`` Dice.
    """
    if label_ids is None:
        label_ids = list(SYNTHSEG_LABEL_MAP.values())

    real_data = np.asarray(nib.load(str(real_label_path)).dataobj, dtype=np.int32)
    gen_data = np.asarray(nib.load(str(gen_label_path)).dataobj, dtype=np.int32)

    dices: dict[str, float] = {}
    for name, lid in SYNTHSEG_LABEL_MAP.items():
        if lid not in label_ids:
            continue
        real_mask = real_data == lid
        gen_mask = gen_data == lid
        intersection = np.sum(real_mask & gen_mask)
        denom = np.sum(real_mask) + np.sum(gen_mask)
        if denom == 0:
            dices[name] = 1.0  # Both empty -> perfect match
        else:
            dices[name] = 2.0 * intersection / denom

    if dices:
        dices["mean"] = float(np.mean(list(dices.values())))

    return dices


def compute_regional_correlation(
    real_volumes: dict[str, dict[str, float]],
    gen_volumes: dict[str, dict[str, float]],
    nn_indices: Tensor | None = None,
    regions: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-region Pearson correlation between real and generated volumes.

    Args:
        real_volumes: Dict ``{subject: {region: volume_mm3}}`` for real volumes.
        gen_volumes: Dict ``{subject: {region: volume_mm3}}`` for generated.
        nn_indices: NN pairing indices ``(n_gen,)`` mapping gen[i] to
            real[nn_indices[i]].  If None, pairs by position.
        regions: Region names to evaluate.  If None, uses all from
            ``SYNTHSEG_LABEL_MAP``.

    Returns:
        Dict ``{region: {"pearson_r": float, "p_value": float}}``.
    """
    from scipy.stats import pearsonr

    if regions is None:
        regions = list(SYNTHSEG_LABEL_MAP.keys())

    real_keys = sorted(real_volumes.keys())
    gen_keys = sorted(gen_volumes.keys())

    results: dict[str, dict[str, float]] = {}

    for region in regions:
        real_vals: list[float] = []
        gen_vals: list[float] = []

        for i, gk in enumerate(gen_keys):
            if region not in gen_volumes[gk]:
                continue

            if nn_indices is not None:
                real_idx = int(nn_indices[i].item()) if i < len(nn_indices) else i
            else:
                real_idx = i

            if real_idx >= len(real_keys):
                continue

            rk = real_keys[real_idx]
            if region not in real_volumes[rk]:
                continue

            real_vals.append(real_volumes[rk][region])
            gen_vals.append(gen_volumes[gk][region])

        if len(real_vals) < 3:
            results[region] = {"pearson_r": float("nan"), "p_value": float("nan")}
            continue

        r, p = pearsonr(real_vals, gen_vals)
        results[region] = {"pearson_r": float(r), "p_value": float(p)}

    return results


def compute_regional_kl(
    real_volumes: dict[str, dict[str, float]],
    gen_volumes: dict[str, dict[str, float]],
    regions: list[str] | None = None,
    n_bins: int = 50,
) -> dict[str, float]:
    """Compute per-region KL divergence of volume histograms.

    Args:
        real_volumes: Dict ``{subject: {region: volume_mm3}}`` for real volumes.
        gen_volumes: Dict ``{subject: {region: volume_mm3}}`` for generated.
        regions: Region names.  If None, uses all from ``SYNTHSEG_LABEL_MAP``.
        n_bins: Number of histogram bins.

    Returns:
        Dict ``{region: kl_divergence}``.
    """
    if regions is None:
        regions = list(SYNTHSEG_LABEL_MAP.keys())

    real_keys = sorted(real_volumes.keys())
    gen_keys = sorted(gen_volumes.keys())

    results: dict[str, float] = {}

    for region in regions:
        real_vals = [real_volumes[rk][region] for rk in real_keys if region in real_volumes[rk]]
        gen_vals = [gen_volumes[gk][region] for gk in gen_keys if region in gen_volumes[gk]]

        if len(real_vals) < 5 or len(gen_vals) < 5:
            results[region] = float("nan")
            continue

        all_vals = real_vals + gen_vals
        if abs(max(all_vals) - min(all_vals)) < 1e-10:
            results[region] = float("nan")
            continue
        bins = np.linspace(min(all_vals), max(all_vals), n_bins + 1)

        real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_vals, bins=bins, density=True)

        eps = 1e-10
        real_hist = real_hist + eps
        gen_hist = gen_hist + eps
        real_hist = real_hist / real_hist.sum()
        gen_hist = gen_hist / gen_hist.sum()

        kl = float(np.sum(real_hist * np.log(real_hist / gen_hist)))
        results[region] = kl

    return results


def consolidate_nifti_to_h5(
    nifti_dir: Path,
    output_h5: Path,
    delete_nifti: bool = False,
) -> Path:
    """Pack NIfTI label maps (or volumes) into a single HDF5 archive.

    Consolidates many small NIfTI files into one compressed HDF5 to reduce
    inode usage on HPC filesystems.

    Args:
        nifti_dir: Directory containing ``.nii.gz`` files.
        output_h5: Output HDF5 path.
        delete_nifti: If True, delete ``nifti_dir`` after successful consolidation.

    Returns:
        Path to the created HDF5 file.
    """
    nifti_files = sorted(nifti_dir.glob("*.nii.gz"))
    if not nifti_files:
        logger.warning("No NIfTI files found in %s — skipping consolidation", nifti_dir)
        return output_h5

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # Read first file to determine shape/dtype
    first_data = np.asarray(nib.load(str(nifti_files[0])).dataobj)
    vol_shape = first_data.shape
    # Use int16 for label maps (integer data), float32 for volumes
    is_integer = np.issubdtype(first_data.dtype, np.integer)
    store_dtype = np.int16 if is_integer else np.float32

    n = len(nifti_files)
    filenames = [f.name for f in nifti_files]

    with h5py.File(str(output_h5), "w") as f:
        ds = f.create_dataset(
            "data",
            shape=(n, *vol_shape),
            dtype=store_dtype,
            chunks=(1, *vol_shape),
            compression="gzip",
            compression_opts=4,
        )

        # Store first volume (already loaded)
        ds[0] = first_data.astype(store_dtype)

        for i in range(1, n):
            data = np.asarray(nib.load(str(nifti_files[i])).dataobj)
            ds[i] = data.astype(store_dtype)

        # Store filenames for traceability
        dt = h5py.string_dtype()
        f.create_dataset("filenames", data=filenames, dtype=dt)
        f.attrs["n_files"] = n
        f.attrs["source_dir"] = str(nifti_dir)

    logger.info(
        "Consolidated %d NIfTI files from %s → %s",
        n,
        nifti_dir.name,
        output_h5.name,
    )

    if delete_nifti:
        shutil.rmtree(nifti_dir)
        logger.info("Deleted NIfTI directory: %s", nifti_dir)

    return output_h5


def run_synthseg_evaluation(
    gen_volumes_h5: Path,
    real_volumes_h5: Path,
    output_dir: Path,
    nn_indices: Tensor | None = None,
    config: SynthSegConfig | None = None,
    nfe: int = 0,
) -> dict | None:
    """Run full SynthSeg evaluation pipeline.

    Converts HDF5 volumes to NIfTI, runs SynthSeg on both real and generated,
    computes all morphological metrics.

    Args:
        gen_volumes_h5: Path to generated volumes HDF5.
        real_volumes_h5: Path to real test volumes HDF5.
        output_dir: Base output directory for SynthSeg results.
        nn_indices: NN pairing indices for paired metrics.
        config: SynthSeg configuration.
        nfe: NFE level (for directory naming).

    Returns:
        Dict with morphological metrics, or None if SynthSeg is unavailable.
    """
    if config is None:
        config = SynthSegConfig()

    if not check_synthseg_available(config):
        logger.warning(
            "SynthSeg not available (command=%s). Skipping morphological evaluation.",
            config.command,
        )
        return None

    synthseg_dir = output_dir / "synthseg"

    # Directories for NIfTI conversion and SynthSeg output
    real_nifti_dir = synthseg_dir / "real_nifti"
    gen_nifti_dir = synthseg_dir / f"gen_nifti_nfe{nfe:03d}"
    real_labels_dir = synthseg_dir / "real_labels"
    gen_labels_dir = synthseg_dir / f"gen_labels_nfe{nfe:03d}"
    real_vol_csv = synthseg_dir / "real_volumes.csv"
    gen_vol_csv = synthseg_dir / f"gen_volumes_nfe{nfe:03d}.csv"
    gen_qc_csv = synthseg_dir / f"gen_qc_nfe{nfe:03d}.csv"

    # Convert HDF5 -> NIfTI
    logger.info("Converting real volumes to NIfTI...")
    _h5_volumes_to_nifti(real_volumes_h5, real_nifti_dir, config.voxel_size_mm)

    logger.info("Converting generated volumes to NIfTI...")
    _h5_volumes_to_nifti(gen_volumes_h5, gen_nifti_dir, config.voxel_size_mm)

    # Run SynthSeg on real (skip if already done — shared across NFE levels)
    if not real_vol_csv.exists():
        logger.info("Running SynthSeg on real volumes...")
        ok = run_synthseg(real_nifti_dir, real_labels_dir, real_vol_csv, config)
        if not ok:
            logger.error("SynthSeg failed on real volumes")
            return None
    else:
        logger.info("Real SynthSeg results already cached, reusing.")

    # Run SynthSeg on generated
    if not gen_vol_csv.exists():
        logger.info("Running SynthSeg on generated volumes (NFE=%d)...", nfe)
        ok = run_synthseg(
            gen_nifti_dir,
            gen_labels_dir,
            gen_vol_csv,
            config,
            qc_csv=gen_qc_csv,
        )
        if not ok:
            logger.error("SynthSeg failed on generated volumes (NFE=%d)", nfe)
            return None
    else:
        logger.info("Generated SynthSeg results (NFE=%d) already cached, reusing.", nfe)

    # --- Compute metrics ---
    results: dict = {}

    # Success rate
    gen_success = compute_success_rate(gen_nifti_dir, gen_labels_dir)
    results["synthseg_success_rate"] = gen_success

    # Parse volume CSVs
    real_vols = parse_volumes_csv(real_vol_csv)
    gen_vols = parse_volumes_csv(gen_vol_csv)

    # Validate CSV outputs (catch blank-data failures)
    real_valid, real_msg = validate_volumes_csv(real_vols)
    if not real_valid:
        logger.error("Real volumes CSV validation FAILED: %s", real_msg)
        results["validation"] = f"FAILED (real): {real_msg}"

    gen_valid, gen_msg = validate_volumes_csv(gen_vols)
    if not gen_valid:
        logger.error("Generated volumes CSV validation FAILED: %s", gen_msg)
        results["validation"] = results.get("validation", "") + f"FAILED (gen): {gen_msg}"

    # Regional volume correlation (paired via NN)
    results["regional_volume_correlation"] = compute_regional_correlation(
        real_vols,
        gen_vols,
        nn_indices,
        config.regions_of_interest,
    )

    # Regional volume KL (distributional, unpaired)
    results["regional_volume_kl"] = compute_regional_kl(
        real_vols,
        gen_vols,
        config.regions_of_interest,
    )

    # Dice overlap (on paired volumes via NN)
    dice_scores: list[float] = []
    gen_label_files = sorted(gen_labels_dir.glob("*.nii.gz"))
    real_label_files = sorted(real_labels_dir.glob("*.nii.gz"))

    for i, gen_lbl in enumerate(gen_label_files):
        if nn_indices is not None and i < len(nn_indices):
            real_idx = int(nn_indices[i].item())
        else:
            real_idx = i

        if real_idx >= len(real_label_files):
            continue

        real_lbl = real_label_files[real_idx]
        dices = compute_dice_from_labels(real_lbl, gen_lbl)
        if "mean" in dices:
            dice_scores.append(dices["mean"])

    if dice_scores:
        dice_arr = np.array(dice_scores)
        results["synthseg_dice"] = {
            "mean": float(dice_arr.mean()),
            "std": float(dice_arr.std()),
            "n_pairs": len(dice_scores),
        }
    else:
        results["synthseg_dice"] = {
            "mean": float("nan"),
            "std": float("nan"),
            "n_pairs": 0,
        }

    logger.info(
        "SynthSeg evaluation (NFE=%d): success=%.2f%%, Dice=%.4f +/- %.4f",
        nfe,
        gen_success * 100,
        results["synthseg_dice"]["mean"],
        results["synthseg_dice"]["std"],
    )

    # Consolidate NIfTI files → HDF5 to reduce inode usage on HPC
    for nifti_dir, h5_name in [
        (real_nifti_dir, "real_nifti.h5"),
        (gen_nifti_dir, f"gen_nifti_nfe{nfe:03d}.h5"),
        (real_labels_dir, "real_labels.h5"),
        (gen_labels_dir, f"gen_labels_nfe{nfe:03d}.h5"),
    ]:
        if nifti_dir.is_dir() and any(nifti_dir.glob("*.nii.gz")):
            consolidate_nifti_to_h5(
                nifti_dir,
                synthseg_dir / h5_name,
                delete_nifti=True,
            )

    return results
