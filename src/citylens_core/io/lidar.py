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
        xs = np.asarray(las.x)
        ys = np.asarray(las.y)
        zs = np.asarray(las.z)
    except Exception:
        return base_height, mask_arr, "mask"

    if xs.size == 0:
        return base_height, mask_arr, "mask"

    inv = ~transform
    height_map = np.full(mask_arr.shape, np.nan, dtype=np.float32)
    coverage_mask = np.zeros(mask_arr.shape, dtype=bool)

    for x, y, z in zip(xs, ys, zs):
        col, row = inv * (float(x), float(y))
        c = int(col)
        r = int(row)
        if 0 <= r < mask_arr.shape[0] and 0 <= c < mask_arr.shape[1] and mask_arr[r, c]:
            coverage_mask[r, c] = True
            existing = height_map[r, c]
            if np.isnan(existing) or float(z) > existing:
                height_map[r, c] = float(z)

    if np.isfinite(height_map).any():
        return np.where(np.isfinite(height_map), height_map, base_height), coverage_mask, "lidar"

    return base_height, mask_arr, "mask"
