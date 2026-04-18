from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_height_map_from_lidar"]


def build_height_map_from_lidar(
    mask: Any,
    lidar_path: Path,
    transform: Any | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a reconstruction height map using LiDAR when available.

    Returns:
        height_map: float32 grid used by the mesh writer
        footprint_mask: bool grid showing LiDAR coverage when LiDAR was used,
            otherwise the mask footprint
        source: "lidar" when LiDAR contributed, otherwise "mask"
    """

    mask_arr = np.asarray(mask).astype(bool)
    base_height = mask_arr.astype(np.float32)

    lidar_path = Path(lidar_path)
    if not lidar_path.exists() or transform is None:
        return base_height, mask_arr, "mask"

    try:
        import laspy
    except Exception:
        return base_height, mask_arr, "mask"

    try:
        las = laspy.read(str(lidar_path))
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)
        zs = np.asarray(las.z, dtype=np.float32)
    except Exception:
        return base_height, mask_arr, "mask"

    if xs.size == 0:
        return base_height, mask_arr, "mask"

    # Vectorized inverse-affine: world (x,y) -> raster (col,row).
    # Affine transforms from `rasterio.transform.Affine` expose .a..f coefficients
    # where:  x = a*col + b*row + c  /  y = d*col + e*row + f.
    # The inverse is another affine; we compute it element-wise across numpy arrays.
    inv = ~transform
    cols_f = inv.a * xs + inv.b * ys + inv.c
    rows_f = inv.d * xs + inv.e * ys + inv.f
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)

    h, w = mask_arr.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(in_bounds):
        return base_height, mask_arr, "mask"

    rows_ib = rows[in_bounds]
    cols_ib = cols[in_bounds]
    zs_ib = zs[in_bounds]

    # Only keep points that fall inside the segmentation footprint.
    inside = mask_arr[rows_ib, cols_ib]
    if not np.any(inside):
        return base_height, mask_arr, "mask"

    rows_in = rows_ib[inside]
    cols_in = cols_ib[inside]
    zs_in = zs_ib[inside]

    # Per-cell max-z. `np.maximum.at` is the unbuffered scatter op that handles
    # duplicate indices correctly (plain assignment would drop all but one).
    height_map = np.full(mask_arr.shape, -np.inf, dtype=np.float32)
    np.maximum.at(height_map, (rows_in, cols_in), zs_in)

    finite = np.isfinite(height_map)
    if not np.any(finite):
        return base_height, mask_arr, "mask"

    coverage_mask = finite
    blended = np.where(finite, height_map, base_height).astype(np.float32)
    return blended, coverage_mask, "lidar"
