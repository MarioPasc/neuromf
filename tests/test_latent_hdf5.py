"""Unit tests for the HDF5 latent shard I/O module.

Verifies create/write/read round-trip, resume semantics, multi-shard
global index, and written mask tracking.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromf.data.latent_hdf5 import (
    build_global_index,
    create_shard,
    discover_shards,
    get_written_mask,
    read_sample,
    write_sample,
)


@pytest.mark.phase1
@pytest.mark.critical
def test_create_and_read_shard(tmp_path: Path) -> None:
    """Create a shard, write samples, read back and verify values match."""
    shard_path = tmp_path / "TEST_DS.h5"
    n_vols = 5
    shape = (4, 8, 8, 8)

    f = create_shard(shard_path, "TEST_DS", n_vols, latent_shape=shape)

    # Write all samples
    tensors = []
    for i in range(n_vols):
        z = torch.randn(*shape)
        tensors.append(z)
        write_sample(f, i, z, f"sub_{i:03d}", f"ses_{i}", f"/path/to/vol_{i}.nii.gz")

    f.close()

    # Read back
    import h5py

    with h5py.File(str(shard_path), "r") as rf:
        assert rf.attrs["dataset_name"] == "TEST_DS"
        assert rf.attrs["n_volumes"] == n_vols
        assert tuple(rf.attrs["latent_shape"]) == shape

        for i in range(n_vols):
            z_read, meta = read_sample(rf, i)
            assert z_read.shape == shape
            assert torch.allclose(z_read, tensors[i], atol=1e-6)
            assert meta["subject_id"] == f"sub_{i:03d}"
            assert meta["session_id"] == f"ses_{i}"
            assert meta["source_path"] == f"/path/to/vol_{i}.nii.gz"
            assert meta["dataset"] == "TEST_DS"


@pytest.mark.phase1
@pytest.mark.critical
def test_resume_encoding(tmp_path: Path) -> None:
    """Partial write then resume: check written mask and resume fills gaps."""
    shard_path = tmp_path / "RESUME.h5"
    n_vols = 10
    shape = (4, 8, 8, 8)

    # Phase 1: write first 6 samples
    f = create_shard(shard_path, "RESUME_DS", n_vols, latent_shape=shape)
    for i in range(6):
        write_sample(f, i, torch.randn(*shape), f"sub_{i}", f"ses_{i}", f"/p/{i}")
    f.close()

    # Check written mask
    import h5py

    with h5py.File(str(shard_path), "r") as rf:
        mask = get_written_mask(rf)
        assert mask[:6].all()
        assert not mask[6:].any()
        assert mask.sum() == 6

    # Phase 2: resume — write remaining 4
    with h5py.File(str(shard_path), "a") as af:
        mask = get_written_mask(af)
        for i in range(n_vols):
            if not mask[i]:
                write_sample(af, i, torch.randn(*shape), f"sub_{i}", f"ses_{i}", f"/p/{i}")

    # Verify all written
    with h5py.File(str(shard_path), "r") as rf:
        mask = get_written_mask(rf)
        assert mask.all()


@pytest.mark.phase1
@pytest.mark.critical
def test_build_global_index_multi_shard(tmp_path: Path) -> None:
    """3 shards with varying sizes: verify flat global index is correct."""
    shape = (4, 8, 8, 8)
    shard_sizes = [3, 5, 2]
    shard_written = [3, 3, 2]  # not all slots written in shard 1

    for idx, (n_total, n_write) in enumerate(zip(shard_sizes, shard_written)):
        shard_path = tmp_path / f"DS_{idx}.h5"
        f = create_shard(shard_path, f"DS_{idx}", n_total, latent_shape=shape)
        for i in range(n_write):
            write_sample(f, i, torch.randn(*shape), f"sub_{i}", f"ses_{i}", f"/p/{i}")
        f.close()

    shards = discover_shards(tmp_path)
    assert len(shards) == 3

    global_idx = build_global_index(shards)
    expected_total = sum(shard_written)
    assert len(global_idx) == expected_total

    # Verify structure
    for shard_path, local_idx in global_idx:
        assert shard_path.exists()
        assert isinstance(local_idx, int)
        assert local_idx >= 0


@pytest.mark.phase1
@pytest.mark.critical
def test_written_mask_tracks_correctly(tmp_path: Path) -> None:
    """Write a subset of slots and verify written mask is precise."""
    shard_path = tmp_path / "MASK_TEST.h5"
    n_vols = 8
    shape = (4, 8, 8, 8)

    f = create_shard(shard_path, "MASK_DS", n_vols, latent_shape=shape)

    # Write only even indices
    written_indices = [0, 2, 4, 6]
    for i in written_indices:
        write_sample(f, i, torch.randn(*shape), f"sub_{i}", f"ses_{i}", f"/p/{i}")

    mask = get_written_mask(f)
    f.close()

    assert mask.sum() == len(written_indices)
    for i in range(n_vols):
        if i in written_indices:
            assert mask[i], f"Index {i} should be written"
        else:
            assert not mask[i], f"Index {i} should NOT be written"


@pytest.mark.phase1
@pytest.mark.informational
def test_discover_shards_empty_dir(tmp_path: Path) -> None:
    """discover_shards returns empty list for directory with no .h5 files."""
    assert discover_shards(tmp_path) == []
