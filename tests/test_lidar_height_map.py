from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from affine import Affine

from citylens_core.io.lidar import build_height_map_from_lidar


def _install_fake_laspy(monkeypatch, xs, ys, zs) -> None:
    fake_module = SimpleNamespace(
        read=lambda path: SimpleNamespace(
            x=np.asarray(xs),
            y=np.asarray(ys),
            z=np.asarray(zs),
        )
    )
    monkeypatch.setitem(sys.modules, "laspy", fake_module)


def test_vectorized_max_z_with_duplicate_points(tmp_path: Path, monkeypatch) -> None:
    las_path = tmp_path / "lidar.las"
    las_path.write_bytes(b"fake")

    # Two points land in cell (0,0); max-z must win.
    _install_fake_laspy(
        monkeypatch,
        xs=[0.1, 0.3, 1.1, 1.1],
        ys=[0.1, 0.2, 0.1, 1.1],
        zs=[5.0, 15.0, 20.0, 40.0],
    )

    mask = np.ones((2, 2), dtype=bool)
    height_map, coverage, source = build_height_map_from_lidar(
        mask, las_path, Affine.identity()
    )

    assert source == "lidar"
    assert coverage.shape == (2, 2)
    assert coverage[0, 0] is np.True_ or bool(coverage[0, 0]) is True
    # cell (0,0) had two points with z=5 and z=15 -> max = 15
    assert height_map[0, 0] == np.float32(15.0)
    # single-point cells keep their z
    assert height_map[0, 1] == np.float32(20.0)
    assert height_map[1, 1] == np.float32(40.0)
    # cell (1,0) had no points -> falls back to base_height (1.0 from mask)
    assert height_map[1, 0] == np.float32(1.0)


def test_out_of_bounds_points_are_ignored(tmp_path: Path, monkeypatch) -> None:
    las_path = tmp_path / "lidar.las"
    las_path.write_bytes(b"fake")

    # Only one in-bounds point; the rest are outside the 2x2 grid.
    _install_fake_laspy(
        monkeypatch,
        xs=[-5.0, 10.0, 0.5, 7.0],
        ys=[-5.0, 10.0, 0.5, -2.0],
        zs=[99.0, 99.0, 7.0, 99.0],
    )

    mask = np.ones((2, 2), dtype=bool)
    height_map, coverage, source = build_height_map_from_lidar(
        mask, las_path, Affine.identity()
    )

    assert source == "lidar"
    assert coverage[0, 0] == True  # noqa: E712
    assert height_map[0, 0] == np.float32(7.0)
    # The other cells had no in-bounds points.
    assert not coverage[0, 1]
    assert not coverage[1, 0]
    assert not coverage[1, 1]


def test_points_outside_mask_footprint_are_ignored(tmp_path: Path, monkeypatch) -> None:
    las_path = tmp_path / "lidar.las"
    las_path.write_bytes(b"fake")

    _install_fake_laspy(
        monkeypatch,
        xs=[0.1, 1.1],
        ys=[0.1, 0.1],
        zs=[12.0, 34.0],
    )

    # Only cell (0,1) is inside the footprint.
    mask = np.array([[False, True], [False, False]], dtype=bool)
    height_map, coverage, source = build_height_map_from_lidar(
        mask, las_path, Affine.identity()
    )

    assert source == "lidar"
    assert coverage[0, 1]
    assert height_map[0, 1] == np.float32(34.0)
    assert not coverage[0, 0]


def test_no_laspy_falls_back_to_mask(tmp_path: Path, monkeypatch) -> None:
    las_path = tmp_path / "lidar.las"
    las_path.write_bytes(b"fake")
    # Simulate laspy not being importable.
    monkeypatch.setitem(sys.modules, "laspy", None)

    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    height_map, coverage, source = build_height_map_from_lidar(
        mask, las_path, Affine.identity()
    )

    assert source == "mask"
    assert height_map.dtype == np.float32
    # base_height is mask-as-float, coverage is the mask itself
    assert np.array_equal(coverage, mask)
    assert height_map[0, 0] == np.float32(1.0)
    assert height_map[0, 1] == np.float32(0.0)


def test_missing_lidar_file_falls_back_to_mask(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.las"
    mask = np.ones((2, 2), dtype=bool)
    _, coverage, source = build_height_map_from_lidar(mask, missing, Affine.identity())
    assert source == "mask"
    assert np.array_equal(coverage, mask)


def test_large_point_cloud_completes_quickly(tmp_path: Path, monkeypatch) -> None:
    """Regression test: the previous per-point Python loop was O(N) with a heavy
    constant and made ~1M-point tiles take minutes. The vectorized version
    should chew through 1M points in well under a second."""

    las_path = tmp_path / "lidar.las"
    las_path.write_bytes(b"fake")

    n = 1_000_000
    rng = np.random.default_rng(42)
    _install_fake_laspy(
        monkeypatch,
        xs=rng.uniform(0, 256, size=n),
        ys=rng.uniform(0, 256, size=n),
        zs=rng.uniform(0, 100, size=n).astype(np.float32),
    )

    mask = np.ones((256, 256), dtype=bool)

    import time

    t0 = time.perf_counter()
    height_map, coverage, source = build_height_map_from_lidar(
        mask, las_path, Affine.identity()
    )
    elapsed = time.perf_counter() - t0

    assert source == "lidar"
    assert coverage.any()
    # Generous upper bound; in practice this runs in ~50-200ms on modest CPUs.
    assert elapsed < 5.0, f"vectorized path too slow: {elapsed:.2f}s for {n} points"
