"""Unit tests for citylens_core.stages._registration."""

from __future__ import annotations

import numpy as np
import pytest

from citylens_core.stages._registration import (
    RegistrationResult,
    apply_shift,
    estimate_alignment,
)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def _make_building_mask(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, h, w in rects:
        r1 = min(shape[0], r0 + h)
        c1 = min(shape[1], c0 + w)
        r0c = max(0, r0)
        c0c = max(0, c0)
        if r1 > r0c and c1 > c0c:
            mask[r0c:r1, c0c:c1] = True
    return mask


def _shift_mask_int(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        out[:dy, :] = False
    elif dy < 0:
        out[dy:, :] = False
    if dx > 0:
        out[:, :dx] = False
    elif dx < 0:
        out[:, dx:] = False
    return out


def test_identity_case() -> None:
    """baseline == current: shift is (0, 0), no IoU gain, not accepted."""
    rng = np.random.default_rng(0)
    mask = np.zeros((128, 128), dtype=bool)
    for _ in range(5):
        r = int(rng.integers(10, 110))
        c = int(rng.integers(10, 110))
        h = int(rng.integers(8, 18))
        w = int(rng.integers(8, 18))
        mask[r : r + h, c : c + w] = True

    result = estimate_alignment(mask, mask)

    assert isinstance(result, RegistrationResult)
    assert result.dy == 0.0
    assert result.dx == 0.0
    assert result.iou_before == pytest.approx(1.0)
    assert result.iou_after == pytest.approx(1.0)
    assert result.accepted is False
    assert result.confidence > 0.5


def test_pure_translation_recovers_shift() -> None:
    shape = (256, 256)
    current = _make_building_mask(shape, [(100, 100, 30, 30)])
    baseline = _shift_mask_int(current, 2, 3)

    result = estimate_alignment(baseline, current)

    assert int(result.dy) == -2
    assert int(result.dx) == -3
    assert result.confidence >= 0.15

    aligned = apply_shift(baseline, dy=result.dy, dx=result.dx)
    assert _iou(aligned, current) == pytest.approx(1.0, abs=1e-9)
    assert result.iou_after > result.iou_before
    assert result.accepted is True


def test_sub_pixel_shift_is_recovered_to_nearest_int() -> None:
    shape = (256, 256)

    def rasterize_rect(cy: float, cx: float, h: float, w: float) -> np.ndarray:
        ys, xs = np.indices(shape)
        return (
            (ys >= cy - h / 2)
            & (ys < cy + h / 2)
            & (xs >= cx - w / 2)
            & (xs < cx + w / 2)
        )

    current = rasterize_rect(120.0, 130.0, 30.0, 30.0)
    baseline = rasterize_rect(121.5, 131.5, 30.0, 30.0)

    result = estimate_alignment(baseline, current)

    assert int(result.dy) in (-1, -2)
    assert int(result.dx) in (-1, -2)
    assert result.iou_after - result.iou_before > 0.05
    assert result.accepted is True


def test_empty_mask_returns_noop() -> None:
    shape = (64, 64)
    baseline = np.zeros(shape, dtype=bool)
    current = _make_building_mask(shape, [(20, 20, 10, 10)])

    result = estimate_alignment(baseline, current)

    assert result.dy == 0.0
    assert result.dx == 0.0
    assert result.confidence == 0.0
    assert result.accepted is False
    assert result.iou_before == result.iou_after

    result2 = estimate_alignment(current, np.zeros(shape, dtype=bool))
    assert result2.confidence == 0.0
    assert result2.accepted is False


def test_random_noise_is_rejected() -> None:
    rng = np.random.default_rng(42)
    shape = (128, 128)
    a = (rng.integers(0, 2, size=shape)).astype(np.uint8)
    b = (rng.integers(0, 2, size=shape)).astype(np.uint8)

    result = estimate_alignment(a, b, min_confidence=0.15, min_iou_gain=0.01)

    assert result.accepted is False


def test_max_shift_cap_clips_large_translation() -> None:
    shape = (256, 256)
    current = _make_building_mask(shape, [(100, 100, 30, 30)])
    baseline = _shift_mask_int(current, 10, 10)

    result = estimate_alignment(baseline, current, max_shift_px=4)

    assert abs(int(result.dy)) <= 4
    assert abs(int(result.dx)) <= 4
    assert result.iou_after < 0.9


def test_realistic_nyc_grid_recovers_per_building_iou() -> None:
    rng = np.random.default_rng(2024)
    shape = (1024, 1024)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(30):
        h = int(rng.integers(6, 40))
        w = int(rng.integers(6, 40))
        r0 = int(rng.integers(20, shape[0] - h - 20))
        c0 = int(rng.integers(20, shape[1] - w - 20))
        rects.append((r0, c0, h, w))

    current = _make_building_mask(shape, rects)
    baseline = _shift_mask_int(current, 1, -2)

    iou_before_global = _iou(baseline, current)

    result = estimate_alignment(baseline, current)

    assert int(result.dy) == -1
    assert int(result.dx) == 2
    assert result.accepted is True

    aligned = apply_shift(baseline, dy=result.dy, dx=result.dx)
    iou_after_global = _iou(aligned, current)
    assert iou_after_global > iou_before_global
    assert iou_after_global == pytest.approx(1.0, abs=1e-9)


def test_apply_shift_rounds_subpixel_input() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True

    out = apply_shift(mask, dy=1.4, dx=-1.6)
    expected = np.zeros((10, 10), dtype=bool)
    expected[6, 3] = True
    np.testing.assert_array_equal(out, expected)


def test_apply_shift_zeros_wraparound_pixels() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :] = True
    out = apply_shift(mask, dy=-2, dx=0)
    assert not out[0, :].any()
    assert not out[-2:, :].any()


def test_input_is_not_mutated() -> None:
    shape = (64, 64)
    current = _make_building_mask(shape, [(20, 20, 12, 12)])
    baseline = _shift_mask_int(current, 1, 1)
    base_copy = baseline.copy()
    cur_copy = current.copy()

    _ = estimate_alignment(baseline, current)
    np.testing.assert_array_equal(baseline, base_copy)
    np.testing.assert_array_equal(current, cur_copy)

    _ = apply_shift(baseline, dy=1, dx=1)
    np.testing.assert_array_equal(baseline, base_copy)
