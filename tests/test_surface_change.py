"""Unit tests for citylens_core.stages._surface_change."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from citylens_core.stages._surface_change import (
    SurfaceImages,
    _erode_binary,
    _srgb_to_lab,
    is_surface_changed,
    load_surface_images,
    surface_delta_e,
)


def _make_images(current: np.ndarray, baseline: np.ndarray) -> SurfaceImages:
    return SurfaceImages(
        current_rgb=current.astype(np.uint8),
        baseline_rgb=baseline.astype(np.uint8),
    )


def _solid_rgb(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    return arr


def test_identical_images_zero_delta() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    images = _make_images(img.copy(), img.copy())
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 10:50] = True

    de = surface_delta_e(images=images, footprint_mask=mask, erode_px=1)
    assert de is not None
    assert de < 1e-3
    assert is_surface_changed(de) is False


def test_pure_color_repaint_large_delta() -> None:
    baseline = _solid_rgb(64, 64, (255, 0, 0))
    current = _solid_rgb(64, 64, (0, 0, 255))
    images = _make_images(current, baseline)
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 8:56] = True

    de = surface_delta_e(images=images, footprint_mask=mask, erode_px=1)
    assert de is not None
    assert de > 50.0
    assert is_surface_changed(de, threshold=20.0) is True


def test_subtle_shading_small_delta() -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(80, 200, size=(64, 64, 3), dtype=np.uint8)
    cur = (base.astype(np.float32) * 0.95).clip(0, 255).astype(np.uint8)

    images = _make_images(cur, base)
    mask = np.ones((64, 64), dtype=bool)

    de = surface_delta_e(images=images, footprint_mask=mask, erode_px=1)
    assert de is not None
    assert de < 10.0
    assert is_surface_changed(de, threshold=20.0) is False


def test_edge_erosion_protects_interior() -> None:
    h = w = 32
    base = _solid_rgb(h, w, (120, 120, 120))
    cur = _solid_rgb(h, w, (120, 120, 120))

    base[4, 4:28] = (255, 0, 255)
    base[27, 4:28] = (255, 0, 255)
    base[4:28, 4] = (255, 0, 255)
    base[4:28, 27] = (255, 0, 255)

    mask = np.zeros((h, w), dtype=bool)
    mask[4:28, 4:28] = True

    images = _make_images(cur, base)

    de_no_erode = surface_delta_e(images=images, footprint_mask=mask, erode_px=0)
    de_eroded = surface_delta_e(images=images, footprint_mask=mask, erode_px=1)

    assert de_no_erode is not None and de_eroded is not None
    assert de_eroded < 1e-3
    assert de_eroded <= de_no_erode + 1e-6


def test_empty_mask_returns_none() -> None:
    img = _solid_rgb(16, 16, (200, 100, 50))
    images = _make_images(img, img)
    mask = np.zeros((16, 16), dtype=bool)

    assert surface_delta_e(images=images, footprint_mask=mask) is None


def test_tiny_mask_falls_back_to_unerod() -> None:
    base = _solid_rgb(8, 8, (100, 100, 100))
    cur = _solid_rgb(8, 8, (100, 100, 100))
    images = _make_images(cur, base)

    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True

    de = surface_delta_e(images=images, footprint_mask=mask, erode_px=2)
    assert de is not None
    assert de < 1e-3


def test_shape_mismatch_returns_none() -> None:
    img = _solid_rgb(64, 64, (10, 20, 30))
    images = _make_images(img, img)
    mask = np.ones((32, 32), dtype=bool)

    assert surface_delta_e(images=images, footprint_mask=mask) is None


def test_median_robust_to_outlier() -> None:
    h, w = 10, 10
    base = _solid_rgb(h, w, (128, 128, 128))
    bumped = np.full_like(base, 130)
    cur = bumped.copy()
    cur[0, 0] = (0, 255, 0)

    images = _make_images(cur, base)
    mask = np.ones((h, w), dtype=bool)

    de = surface_delta_e(images=images, footprint_mask=mask, erode_px=0)
    assert de is not None
    assert de < 5.0

    cur_lab = _srgb_to_lab(np.array([[[0, 255, 0]]], dtype=np.uint8))
    base_lab = _srgb_to_lab(np.array([[[128, 128, 128]]], dtype=np.uint8))
    outlier_de = float(np.linalg.norm(cur_lab - base_lab))
    assert outlier_de > 50.0


def test_load_surface_images_missing_file(tmp_path: Path) -> None:
    real = tmp_path / "real.png"
    Image.fromarray(_solid_rgb(16, 16, (10, 20, 30))).save(real)

    missing = tmp_path / "does_not_exist.png"

    assert load_surface_images(orthophoto_path=missing, baseline_path=real) is None
    assert load_surface_images(orthophoto_path=real, baseline_path=missing) is None
    assert load_surface_images(orthophoto_path=None, baseline_path=real) is None
    assert load_surface_images(orthophoto_path=real, baseline_path=None) is None


def test_load_surface_images_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_text('{"type": "FeatureCollection", "features": []}')

    good = tmp_path / "good.png"
    Image.fromarray(_solid_rgb(16, 16, (10, 20, 30))).save(good)

    assert load_surface_images(orthophoto_path=good, baseline_path=bad) is None


def test_load_surface_images_shape_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(_solid_rgb(256, 256, (10, 20, 30))).save(a)
    Image.fromarray(_solid_rgb(256, 256, (10, 20, 30))).save(b)

    out = load_surface_images(
        orthophoto_path=a, baseline_path=b, expected_shape=(128, 128)
    )
    assert out is None


def test_load_surface_images_matching_shape(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(_solid_rgb(64, 64, (10, 20, 30))).save(a)
    Image.fromarray(_solid_rgb(64, 64, (40, 50, 60))).save(b)

    out = load_surface_images(
        orthophoto_path=a, baseline_path=b, expected_shape=(64, 64)
    )
    assert out is not None
    assert out.current_rgb.shape == (64, 64, 3)
    assert out.baseline_rgb.shape == (64, 64, 3)


def test_load_surface_images_image_size_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.fromarray(_solid_rgb(64, 64, (10, 20, 30))).save(a)
    Image.fromarray(_solid_rgb(32, 32, (10, 20, 30))).save(b)

    assert load_surface_images(orthophoto_path=a, baseline_path=b) is None


def test_srgb_to_lab_white() -> None:
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    lab = _srgb_to_lab(white)[0, 0]
    assert lab[0] == pytest.approx(100.0, abs=0.5)
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_srgb_to_lab_black() -> None:
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    lab = _srgb_to_lab(black)[0, 0]
    assert lab[0] == pytest.approx(0.0, abs=0.5)
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_srgb_to_lab_preserves_leading_dims() -> None:
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[..., :] = 200
    lab = _srgb_to_lab(img)
    assert lab.shape == (4, 5, 3)
    assert np.allclose(lab, lab[0, 0], atol=1e-4)


def test_erode_binary_zero_steps_is_copy() -> None:
    m = np.array([[True, False], [False, True]])
    out = _erode_binary(m, steps=0)
    assert out.dtype == bool
    assert np.array_equal(out, m)
    assert out is not m


def test_erode_binary_shrinks_block() -> None:
    m = np.zeros((7, 7), dtype=bool)
    m[1:6, 1:6] = True
    out = _erode_binary(m, steps=1)
    expected = np.zeros((7, 7), dtype=bool)
    expected[2:5, 2:5] = True
    assert np.array_equal(out, expected)


def test_erode_binary_empty_after_too_many_steps() -> None:
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    out = _erode_binary(m, steps=1)
    assert not out.any()


def test_is_surface_changed_none_returns_false() -> None:
    assert is_surface_changed(None) is False


def test_is_surface_changed_threshold() -> None:
    assert is_surface_changed(19.9, threshold=20.0) is False
    assert is_surface_changed(20.0, threshold=20.0) is True
    assert is_surface_changed(50.0, threshold=20.0) is True
