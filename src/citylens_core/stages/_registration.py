"""Sub-pixel-rounded mask alignment via FFT phase correlation.

Estimates a single (dy, dx) integer-pixel translation that aligns a
baseline mask to a current mask. Used by `stage_change` to recover the
0.5-2 px misregistration between baseline and current orthophotos
before computing per-footprint IoU. Pure numpy; no scipy / scikit-image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Translation that best aligns `baseline` to `current` in pixel space."""

    dy: float  # rows; positive = baseline shifts DOWN to align
    dx: float  # cols; positive = baseline shifts RIGHT to align
    confidence: float  # 0..1, peak height vs noise floor of phase-correlation result
    iou_before: float  # IoU baseline <-> current at zero shift
    iou_after: float  # IoU after applying the shift (rounded to int pixels)
    accepted: bool  # True iff confidence is sufficient AND iou_after > iou_before


def _to_bool(mask: np.ndarray) -> np.ndarray:
    """Cast input to a 2-D boolean array (no copy if already bool)."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.dtype == bool:
        return mask
    return mask.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union for two boolean masks of identical shape."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _shift_int(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean mask by integer (dy, dx). Wrap-around is zeroed."""
    if dy == 0 and dx == 0:
        return mask.copy()
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


def estimate_alignment(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    max_shift_px: int = 4,
    min_confidence: float = 0.15,
    min_iou_gain: float = 0.01,
) -> RegistrationResult:
    """Estimate the best (dy, dx) integer-pixel translation that aligns
    `baseline` to `current`, using FFT phase correlation.

    Both inputs are 2-D boolean / uint8 masks of the same shape.

    Returns a RegistrationResult. Caller is expected to apply the shift
    via `apply_shift(...)` only when `result.accepted` is True.
    """
    b = _to_bool(baseline)
    c = _to_bool(current)
    if b.shape != c.shape:
        raise ValueError(f"shape mismatch: {b.shape} vs {c.shape}")

    iou_before = _iou(b, c)

    if not b.any() or not c.any():
        return RegistrationResult(
            dy=0.0,
            dx=0.0,
            confidence=0.0,
            iou_before=iou_before,
            iou_after=iou_before,
            accepted=False,
        )

    h, w = b.shape

    bf = b.astype(np.float64)
    cf = c.astype(np.float64)
    bf = bf - bf.mean()
    cf = cf - cf.mean()

    F = np.fft.fft2(bf)
    G = np.fft.fft2(cf)

    cross = G * np.conj(F)
    mag = np.abs(cross)
    mag = np.where(mag < 1e-12, 1e-12, mag)
    R = cross / mag

    r = np.fft.ifft2(R).real
    r_abs = np.abs(r)

    k = int(max(1, max_shift_px))
    k = min(k, h // 2 - 1, w // 2 - 1)
    if k < 1:
        k = 1

    rows = np.concatenate([np.arange(0, k + 1), np.arange(h - k, h)])
    cols = np.concatenate([np.arange(0, k + 1), np.arange(w - k, w)])
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    window = r_abs[rr, cc]

    flat_idx = int(np.argmax(window))
    pi, pj = np.unravel_index(flat_idx, window.shape)
    peak_row = int(rows[pi])
    peak_col = int(cols[pj])

    dy = peak_row if peak_row <= h // 2 else peak_row - h
    dx = peak_col if peak_col <= w // 2 else peak_col - w

    dy = int(np.clip(dy, -max_shift_px, max_shift_px))
    dx = int(np.clip(dx, -max_shift_px, max_shift_px))

    peak_val = float(window[pi, pj])
    noise_mask = np.ones_like(window, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = pi + di
            jj = pj + dj
            if 0 <= ii < window.shape[0] and 0 <= jj < window.shape[1]:
                noise_mask[ii, jj] = False
    noise_vals = window[noise_mask]
    if noise_vals.size == 0 or peak_val <= 0.0:
        confidence = 0.0
    else:
        noise_floor = float(np.mean(noise_vals))
        if noise_floor <= 0.0:
            confidence = 1.0 if peak_val > 0.0 else 0.0
        else:
            ratio = peak_val / noise_floor
            confidence = float(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0))

    if dy == 0 and dx == 0:
        iou_after = iou_before
    else:
        shifted = _shift_int(b, dy, dx)
        iou_after = _iou(shifted, c)

    accepted = bool(
        confidence >= min_confidence
        and (iou_after - iou_before) >= min_iou_gain
    )

    return RegistrationResult(
        dy=float(dy),
        dx=float(dx),
        confidence=float(confidence),
        iou_before=float(iou_before),
        iou_after=float(iou_after),
        accepted=accepted,
    )


def apply_shift(mask: np.ndarray, *, dy: float, dx: float) -> np.ndarray:
    """Shift a 2-D mask by (dy, dx) integer pixels using `np.roll`,
    zeroing out the pixels that wrap around. Sub-pixel shifts are
    rounded to the nearest integer."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    idy = int(np.rint(dy))
    idx = int(np.rint(dx))
    if idy == 0 and idx == 0:
        return mask.copy()
    out = np.roll(mask, shift=(idy, idx), axis=(0, 1))
    if idy > 0:
        out[:idy, :] = 0
    elif idy < 0:
        out[idy:, :] = 0
    if idx > 0:
        out[:, :idx] = 0
    elif idx < 0:
        out[:, idx:] = 0
    return out
