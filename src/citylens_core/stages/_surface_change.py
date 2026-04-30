"""Surface-change detection via CIE76 Lab Delta-E.

For polygons whose footprint shape didn't change (high IoU between
baseline and current SAM2 masks), this module re-loads the original RGB
imagery and measures the perceptual color difference inside the
footprint. Catches re-roofing, repainting, solar-panel installs — real
visual changes that shape-only segmentation misses.

Pure numpy + Pillow. No skimage/opencv. The sRGB→Lab conversion uses the
standard sRGB EOTF + sRGB→XYZ matrix + D65 reference white.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SurfaceImages:
    """Cached pair of orthophoto + baseline RGB images, both at the same
    H×W as the masks they'll be sampled with. Built once per
    stage_change invocation and reused across many footprints."""

    current_rgb: np.ndarray  # (H, W, 3) uint8
    baseline_rgb: np.ndarray  # (H, W, 3) uint8


def load_surface_images(
    *,
    orthophoto_path: Path | None,
    baseline_path: Path | None,
    expected_shape: tuple[int, int] | None = None,
) -> SurfaceImages | None:
    """Load both images. Returns None if either is missing, can't be
    decoded, or doesn't match the expected mask shape (e.g. baseline
    was a footprint GeoJSON not an actual image — that case is
    legitimate; the caller just skips surface-change checks)."""
    if orthophoto_path is None or baseline_path is None:
        return None

    current = _load_rgb(orthophoto_path)
    if current is None:
        return None

    baseline = _load_rgb(baseline_path)
    if baseline is None:
        return None

    if current.shape != baseline.shape:
        return None

    if expected_shape is not None:
        if current.shape[:2] != tuple(expected_shape):
            return None

    return SurfaceImages(current_rgb=current, baseline_rgb=baseline)


def _load_rgb(path: Path) -> np.ndarray | None:
    """Load an image as (H, W, 3) uint8. Returns None on any failure."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except (OSError, ValueError, Image.UnidentifiedImageError):
        return None

    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    return arr


def surface_delta_e(
    *,
    images: SurfaceImages,
    footprint_mask: np.ndarray,
    erode_px: int = 1,
) -> float | None:
    """Median CIE76 Delta E between baseline and current pixels inside
    the (eroded) footprint mask. Returns None when the comparison can't
    be made (mismatched shapes, empty mask, etc.)."""
    if images is None or footprint_mask is None:
        return None

    cur = images.current_rgb
    base = images.baseline_rgb

    if cur.shape != base.shape:
        return None
    if cur.ndim != 3 or cur.shape[2] != 3:
        return None

    mask = np.asarray(footprint_mask)
    if mask.ndim != 2:
        return None
    if mask.shape != cur.shape[:2]:
        return None

    mask_bool = mask.astype(bool, copy=False)
    if not mask_bool.any():
        return None

    if erode_px and erode_px > 0:
        eroded = _erode_binary(mask_bool, steps=int(erode_px))
        sample_mask = eroded if eroded.any() else mask_bool
    else:
        sample_mask = mask_bool

    if not sample_mask.any():
        return None

    cur_pixels = cur[sample_mask]
    base_pixels = base[sample_mask]

    if cur_pixels.size == 0 or base_pixels.size == 0:
        return None

    cur_lab = _srgb_to_lab(cur_pixels)
    base_lab = _srgb_to_lab(base_pixels)

    diff = cur_lab - base_lab
    delta_e = np.sqrt(np.sum(diff * diff, axis=-1))

    if delta_e.size == 0:
        return None

    return float(np.median(delta_e))


def is_surface_changed(delta_e: float | None, *, threshold: float = 20.0) -> bool:
    """None → False (we couldn't measure). Otherwise threshold compare."""
    if delta_e is None:
        return False
    try:
        return float(delta_e) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _erode_binary(mask: np.ndarray, *, steps: int) -> np.ndarray:
    """Pure-numpy 4-neighbour binary erosion. Boundary pixels erode away."""
    if steps <= 0:
        return mask.astype(bool, copy=True)

    out = mask.astype(bool, copy=True)
    for _ in range(int(steps)):
        if not out.any():
            return out
        up = np.zeros_like(out)
        down = np.zeros_like(out)
        left = np.zeros_like(out)
        right = np.zeros_like(out)
        up[1:, :] = out[:-1, :]
        down[:-1, :] = out[1:, :]
        left[:, 1:] = out[:, :-1]
        right[:, :-1] = out[:, 1:]
        out = out & up & down & left & right
    return out


# sRGB → XYZ matrix (D65), Bradford-adapted, IEC 61966-2-1.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# D65 reference white.
_XN = 0.95047
_YN = 1.00000
_ZN = 1.08883


def _srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) → CIE Lab (..., 3) float32."""
    arr = np.asarray(rgb_u8)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    srgb = arr.astype(np.float64) / 255.0

    threshold = 0.04045
    low = srgb / 12.92
    high = np.power((srgb + 0.055) / 1.055, 2.4)
    linear = np.where(srgb <= threshold, low, high)

    xyz = np.einsum("...j,ij->...i", linear, _SRGB_TO_XYZ)

    xn = xyz[..., 0] / _XN
    yn = xyz[..., 1] / _YN
    zn = xyz[..., 2] / _ZN

    delta = 6.0 / 29.0
    delta3 = delta**3
    inv_3_delta_sq = 1.0 / (3.0 * delta * delta)
    four_29 = 4.0 / 29.0

    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(
            t > delta3,
            np.cbrt(np.maximum(t, 0.0)),
            t * inv_3_delta_sq + four_29,
        )

    fx = _f(xn)
    fy = _f(yn)
    fz = _f(zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)
