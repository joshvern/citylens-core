from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_height_map_from_lidar"]

_logger = logging.getLogger(__name__)


def _parse_las_crs(las: Any) -> Any | None:
    """Best-effort CRS extraction from a laspy `LasData`.

    laspy exposes CRS via header.parse_crs() on newer versions; fall back to
    VLR scanning for older file formats. Returns a pyproj CRS or None.
    """
    try:
        from pyproj import CRS
    except Exception:
        return None

    header = getattr(las, "header", None)
    if header is None:
        return None

    # Newer laspy: header.parse_crs() returns a pyproj CRS or None.
    parse_crs = getattr(header, "parse_crs", None)
    if callable(parse_crs):
        try:
            crs = parse_crs()
            if crs is not None:
                return CRS.from_user_input(crs)
        except Exception:
            pass

    # Fallback: look for WKT/GeoKey VLRs.
    vlrs = getattr(header, "vlrs", None) or []
    for vlr in vlrs:
        record = getattr(vlr, "record_id", None)
        wkt = getattr(vlr, "string", None) or getattr(vlr, "parsed_record", None)
        if record in (2111, 2112) and isinstance(wkt, str) and wkt.strip():
            try:
                return CRS.from_wkt(wkt)
            except Exception:
                continue

    return None


def _maybe_reproject(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    src_crs: Any | None,
    dst_crs: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject (xs, ys) from src_crs to dst_crs if both are known & differ.

    NYS NYC TopoBathymetric LiDAR tiles are delivered in NAD83 / NY Long
    Island ftUS (EPSG:6539), but the worker fetches orthos in EPSG:3857.
    Without this step, applying the ortho's inverse-affine to ft-based LiDAR
    coords lands every point outside bounds and silently falls back to
    mask-heights (producing a flat mesh).
    """
    if src_crs is None or dst_crs is None:
        return xs, ys

    try:
        from pyproj import CRS, Transformer
    except Exception:
        return xs, ys

    try:
        src = CRS.from_user_input(src_crs)
        dst = CRS.from_user_input(dst_crs)
    except Exception:
        return xs, ys

    if src == dst:
        return xs, ys

    try:
        transformer = Transformer.from_crs(src, dst, always_xy=True)
        xs2, ys2 = transformer.transform(xs, ys)
        return np.asarray(xs2, dtype=np.float64), np.asarray(ys2, dtype=np.float64)
    except Exception as e:
        _logger.warning(
            "lidar_reprojection_failed",
            extra={"src_crs": str(src_crs), "dst_crs": str(dst_crs), "error": f"{type(e).__name__}: {e}"},
        )
        return xs, ys


def build_height_map_from_lidar(
    mask: Any,
    lidar_path: Path,
    transform: Any | None,
    *,
    dst_crs: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a reconstruction height map using LiDAR when available.

    Args:
        mask: HxW boolean segmentation footprint.
        lidar_path: path to a .las/.laz file.
        transform: rasterio.Affine mapping ortho-CRS world coords to (col, row).
        dst_crs: CRS the `transform` is in (i.e. the ortho CRS). When given
            and it differs from the LAS file's declared CRS, LiDAR points
            are reprojected before the inverse-affine is applied.

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

    # Reproject (xs, ys) into the ortho's CRS when we know both. This is the
    # common case in production: LAS is in EPSG:6539, ortho is in EPSG:3857.
    src_crs = _parse_las_crs(las)
    xs, ys = _maybe_reproject(xs, ys, src_crs=src_crs, dst_crs=dst_crs)

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
