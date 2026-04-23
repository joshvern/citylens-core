from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_height_map_from_lidar", "sample_lidar_heights"]

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
    debug: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject (xs, ys) from src_crs to dst_crs if both are known & differ.

    NYS NYC TopoBathymetric LiDAR tiles are delivered in NAD83 / NY Long
    Island ftUS (EPSG:2263/6539), but the worker fetches orthos in EPSG:3857.
    Without this step, applying the ortho's inverse-affine to ft-based LiDAR
    coords lands every point outside bounds and silently falls back to
    mask-heights (producing a flat mesh).
    """
    if debug is None:
        debug = {}

    if src_crs is None or dst_crs is None:
        debug["reproject"] = "skipped_missing_crs"
        return xs, ys

    try:
        from pyproj import CRS, Transformer
    except Exception as e:
        debug["reproject"] = f"pyproj_import_failed:{type(e).__name__}"
        return xs, ys

    try:
        src = CRS.from_user_input(src_crs)
        dst = CRS.from_user_input(dst_crs)
    except Exception as e:
        debug["reproject"] = f"crs_parse_failed:{type(e).__name__}"
        return xs, ys

    if src == dst:
        debug["reproject"] = "skipped_same_crs"
        return xs, ys

    try:
        transformer = Transformer.from_crs(src, dst, always_xy=True)
        xs2, ys2 = transformer.transform(xs, ys)
        xs2 = np.asarray(xs2, dtype=np.float64)
        ys2 = np.asarray(ys2, dtype=np.float64)
        finite = np.isfinite(xs2) & np.isfinite(ys2)
        debug["reproject"] = "ok"
        debug["reproject_src"] = src.to_string() if hasattr(src, "to_string") else str(src)
        debug["reproject_dst"] = dst.to_string() if hasattr(dst, "to_string") else str(dst)
        debug["reproject_finite_pts"] = int(finite.sum())
        return xs2, ys2
    except Exception as e:
        debug["reproject"] = f"transform_failed:{type(e).__name__}:{e}"
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
    debug: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a reconstruction height map using LiDAR when available.

    Args:
        mask: HxW boolean segmentation footprint.
        lidar_path: path to a .las/.laz file.
        transform: rasterio.Affine mapping ortho-CRS world coords to (col, row).
        dst_crs: CRS the `transform` is in (i.e. the ortho CRS). When given
            and it differs from the LAS file's declared CRS, LiDAR points
            are reprojected before the inverse-affine is applied.
        debug: optional dict to receive breadcrumbs describing which branch
            ran (fallback_reason, reproject status, counts at each gate).
            Callers (e.g. the pipeline) can stash this into run_summary.json.

    Returns:
        height_map: float32 grid used by the mesh writer
        footprint_mask: bool grid showing LiDAR coverage when LiDAR was used,
            otherwise the mask footprint
        source: "lidar" when LiDAR contributed, otherwise "mask"
    """
    if debug is None:
        debug = {}

    mask_arr = np.asarray(mask).astype(bool)
    base_height = mask_arr.astype(np.float32)
    debug["mask_px"] = int(mask_arr.sum())

    lidar_path = Path(lidar_path)
    if not lidar_path.exists() or transform is None:
        debug["fallback_reason"] = (
            "no_lidar_file" if not lidar_path.exists() else "no_transform"
        )
        return base_height, mask_arr, "mask"

    try:
        import laspy
    except Exception as e:
        debug["fallback_reason"] = f"no_laspy:{type(e).__name__}"
        return base_height, mask_arr, "mask"

    try:
        las = laspy.read(str(lidar_path))
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)
        zs = np.asarray(las.z, dtype=np.float32)
    except Exception as e:
        debug["fallback_reason"] = f"read_failed:{type(e).__name__}:{e}"
        return base_height, mask_arr, "mask"

    if xs.size == 0:
        debug["fallback_reason"] = "empty_point_cloud"
        return base_height, mask_arr, "mask"

    debug["points_total"] = int(xs.size)

    # Reproject (xs, ys) into the ortho's CRS when we know both. This is the
    # common case in production: LAS is in EPSG:2263/6539, ortho is in EPSG:3857.
    src_crs = _parse_las_crs(las)
    debug["src_crs_detected"] = str(src_crs) if src_crs is not None else None
    debug["dst_crs_received"] = str(dst_crs) if dst_crs is not None else None

    xs, ys = _maybe_reproject(xs, ys, src_crs=src_crs, dst_crs=dst_crs, debug=debug)

    # Vectorized inverse-affine: world (x,y) -> raster (col,row).
    inv = ~transform
    cols_f = inv.a * xs + inv.b * ys + inv.c
    rows_f = inv.d * xs + inv.e * ys + inv.f
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)

    h, w = mask_arr.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    in_bounds_n = int(in_bounds.sum())
    debug["points_in_bounds"] = in_bounds_n
    if in_bounds_n == 0:
        debug["fallback_reason"] = "no_points_in_ortho_bbox"
        return base_height, mask_arr, "mask"

    rows_ib = rows[in_bounds]
    cols_ib = cols[in_bounds]
    zs_ib = zs[in_bounds]

    # Only keep points that fall inside the segmentation footprint.
    inside = mask_arr[rows_ib, cols_ib]
    inside_n = int(inside.sum())
    debug["points_inside_mask"] = inside_n
    if inside_n == 0:
        debug["fallback_reason"] = "no_points_inside_segmentation_mask"
        return base_height, mask_arr, "mask"

    rows_in = rows_ib[inside]
    cols_in = cols_ib[inside]
    zs_in = zs_ib[inside]

    # Per-cell max-z.
    height_map = np.full(mask_arr.shape, -np.inf, dtype=np.float32)
    np.maximum.at(height_map, (rows_in, cols_in), zs_in)

    finite = np.isfinite(height_map)
    if not np.any(finite):
        debug["fallback_reason"] = "no_finite_cells_after_scatter"
        return base_height, mask_arr, "mask"

    coverage_mask = finite
    blended = np.where(finite, height_map, base_height).astype(np.float32)
    debug["fallback_reason"] = None
    debug["cells_with_lidar"] = int(coverage_mask.sum())
    debug["z_min"] = float(np.min(height_map[finite]))
    debug["z_max"] = float(np.max(height_map[finite]))
    return blended, coverage_mask, "lidar"


def _las_vertical_unit_to_meters(las: Any) -> float:
    """Multiplier to convert LAS z values into meters.

    NYS NYC TopoBathymetric LiDAR ships in a compound CRS whose vertical
    component is NAVD88 in US survey feet (EPSG:5703 variants). Without
    converting, a 10 m building registers as ~33 m — breaks any height
    threshold that's expressed in meters.

    Best-effort: inspect the compound CRS parsed from the LAS header and
    look at the vertical sub-CRS' axis_info[0].unit_name. Returns 1.0 if
    we can't prove it's feet (i.e. assume meters, the LAS standard default).
    """
    src_crs = _parse_las_crs(las)
    if src_crs is None:
        return 1.0
    try:
        # Compound CRS: pyproj exposes .sub_crs_list with the vertical CRS at index 1.
        sub = getattr(src_crs, "sub_crs_list", None) or []
        for candidate in sub:
            axis_info = getattr(candidate, "axis_info", None) or []
            for axis in axis_info:
                name = (getattr(axis, "unit_name", "") or "").lower()
                if "survey foot" in name or "us survey foot" in name or "ftus" in name:
                    return 0.3048006096012192
                if "foot" in name and "meter" not in name:
                    # International foot (rare but exists in some datasets).
                    return 0.3048
                if "metre" in name or "meter" in name:
                    return 1.0
    except Exception:
        pass
    return 1.0


def sample_lidar_heights(
    lidar_path: Path,
    transform: Any,
    *,
    shape: tuple[int, int],
    dst_crs: Any | None = None,
) -> tuple[np.ndarray, float] | None:
    """Project a LiDAR point cloud onto a pixel grid and return max-z per cell.

    Unlike `build_height_map_from_lidar`, this doesn't take a segmentation
    mask — it returns the full grid so downstream code (change filter,
    per-building extrusion) can query any bbox.

    z values are converted to METERS regardless of the LAS vertical unit
    (NYS LiDAR ships in US survey feet; callers want meters to match the
    pipeline's other m-based thresholds).

    Returns (heights_m, ground_z_m) where:
      - heights_m: float32 array of `shape`, filled with NaN where no LiDAR
        point landed and max-z-in-meters otherwise.
      - ground_z_m: 5th-percentile z (meters) across all in-bounds LiDAR
        points. Ground-plane reference for above-ground-height comparisons.
    Returns None if LiDAR isn't usable (missing file, laspy not installed,
    read failure, zero points, reprojection impossible, etc).
    """
    lidar_path = Path(lidar_path)
    if not lidar_path.exists() or transform is None:
        return None

    try:
        import laspy
    except Exception:
        return None

    try:
        las = laspy.read(str(lidar_path))
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)
        zs = np.asarray(las.z, dtype=np.float32)
    except Exception:
        return None

    if xs.size == 0:
        return None

    src_crs = _parse_las_crs(las)
    xs, ys = _maybe_reproject(xs, ys, src_crs=src_crs, dst_crs=dst_crs)

    # Convert z to meters once, before any downstream comparison.
    z_unit_m = _las_vertical_unit_to_meters(las)
    zs = (zs * np.float32(z_unit_m)).astype(np.float32)

    inv = ~transform
    cols_f = inv.a * xs + inv.b * ys + inv.c
    rows_f = inv.d * xs + inv.e * ys + inv.f
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)

    h, w = shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(in_bounds):
        return None

    rows_ib = rows[in_bounds]
    cols_ib = cols[in_bounds]
    zs_ib = zs[in_bounds]

    heights = np.full(shape, np.nan, dtype=np.float32)
    heights_pos = np.full(shape, -np.inf, dtype=np.float32)
    np.maximum.at(heights_pos, (rows_ib, cols_ib), zs_ib)
    finite = np.isfinite(heights_pos)
    heights[finite] = heights_pos[finite]

    if not np.any(finite):
        return None

    ground_z = float(np.percentile(zs_ib, 5))
    return heights, ground_z
