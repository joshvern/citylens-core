from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


# ----------------------------------------------------------------------
# Tunable thresholds (env-var overridable at deploy time).
# ----------------------------------------------------------------------


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _min_area_m2() -> float:
    """Drop change features smaller than this area in square meters.

    Default 100 m² (roughly one small garage/shed). Kills the sliver-noise
    from imperfect mask alignment at building edges.
    """
    return _float_env("CITYLENS_CHANGE_MIN_AREA_M2", 100.0)


def _unchanged_iou() -> float:
    # Recalibrated against the wider 250m AOI Brooklyn block (169 buildings,
    # see research/change_threshold_calibration.md). The IoU distribution is
    # bimodal with a stable-block peak around 0.40-0.70 and a thin demolished
    # tail near 0. At threshold 0.50 we got 44% "modified" on a block where
    # almost nothing changed 2017→2024 — that's structural SAM2-vs-GDB edge
    # noise, not real modification. At 0.40 we get ~17% modified, which
    # matches what a human reviewer would flag.
    return _float_env("CITYLENS_CHANGE_UNCHANGED_IOU", 0.4)


def _modified_iou() -> float:
    return _float_env("CITYLENS_CHANGE_MODIFIED_IOU", 0.2)


def _added_max_baseline_overlap() -> float:
    """A new component counts as 'added' only if it overlaps ANY baseline
    footprint by less than this fraction of its own area. Catches the case
    where SAM2 traces a building wider than the baseline footprint and
    leaves a sliver along the edge — that sliver should NOT be flagged as
    a new building.
    """
    return _float_env("CITYLENS_CHANGE_ADDED_MAX_BASELINE_OVERLAP", 0.1)


def _added_min_height_m() -> float:
    """LiDAR height gate on 'added' components.

    A candidate new building must rise at least this many meters above the
    ground plane. Kills the most common SAM2 false positives — trees,
    hedges, vehicles, shadows, pavement patterns — without needing a
    separate classifier. Default 2 m (shorter than a 1-story building,
    taller than a van).

    Only applied when the pipeline has a usable LiDAR heights grid.
    Without LiDAR we don't have height information, so all candidate
    components pass this gate (soft fail).
    """
    return _float_env("CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M", 2.0)


def _added_height_percentile() -> float:
    """Percentile of LiDAR z-values within a candidate 'added' footprint to
    compare against the ground plane. Using the 75th percentile (instead
    of the max) makes the gate robust against single tall trees/antennas
    poking through an otherwise-flat patch of grass."""
    return _float_env("CITYLENS_CHANGE_ADDED_HEIGHT_PERCENTILE", 75.0)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _affine_identity():
    from rasterio.transform import Affine

    return Affine.identity()


def _label_components(mask: Any) -> tuple[Any, int]:
    """Return (labels, n) — labels[i,j]==k for cells in component k (1..n).

    Uses rasterio.features.shapes which is already in the core dep tree.
    Same helper as sam2_runner but inlined here to keep the change stage
    dependency graph tight.
    """
    import numpy as np
    from rasterio.features import rasterize, shapes
    from rasterio.transform import Affine

    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int64)
    if not m.any():
        return labels, 0

    m_u8 = m.astype("uint8")
    n = 0
    identity = Affine.identity()
    for geom, value in shapes(m_u8, mask=m, transform=identity):
        if int(value) != 1:
            continue
        n += 1
        comp_mask = rasterize(
            shapes=[(geom, 1)],
            out_shape=(h, w),
            transform=identity,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        comp_mask &= m
        labels[comp_mask] = n
    return labels, n


def _polygon_coords_from_pixel_mask(
    pixel_mask: Any, *, transform: Any | None
) -> list[list[list[list[float]]]]:
    """Trace a boolean mask into a list of GeoJSON Polygon coordinates
    (each item is [outer_ring, hole_1, ...]).
    """
    import numpy as np
    from rasterio.features import shapes

    m = np.asarray(pixel_mask).astype(bool)
    if not m.any():
        return []

    tr = transform if transform is not None else _affine_identity()
    polys: list[list[list[list[float]]]] = []
    m_u8 = m.astype("uint8")
    for geom, value in shapes(m_u8, mask=m, transform=tr):
        if int(value) != 1:
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        polys.append([list(ring) for ring in coords])
    return polys


def _pixel_area_m2(transform: Any | None, crs_value: str | None) -> float:
    """Return the area of one pixel in square meters, or 0 when we can't
    confidently convert (e.g. pixel-space only or unknown CRS)."""
    if transform is None:
        return 0.0
    try:
        # Affine.a is x pixel size; Affine.e is y pixel size (usually negative).
        px = abs(float(transform.a) * float(transform.e))
    except Exception:
        return 0.0
    if not px:
        return 0.0
    s = (crs_value or "").strip().lower()
    # EPSG:3857 has units of meters but scale varies with latitude
    # (the pixel width stored in `a` is already in the projected meters
    # used by rasterio, so px is already m²). For projected CRS in feet,
    # convert.
    if s in ("pixel", ""):
        return 0.0
    if "ftus" in s or "ft" in s.replace("-", "").replace("_", "") and "meter" not in s:
        # 1 US survey foot = 0.3048006096... m, so 1 ft² ≈ 0.09290341 m²
        return px * 0.0929034116
    return px


def _feature(
    *,
    change_type: str,
    coordinates: list[list[list[float]]],
    area_m2: float | None,
    baseline_iou: float | None,
    imagery_year: int,
    baseline_year: int,
    crs_value: str,
    extra_props: dict[str, Any] | None = None,
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "change_type": change_type,
        "imagery_year": imagery_year,
        "baseline_year": baseline_year,
        "crs": crs_value,
    }
    if baseline_iou is not None:
        props["baseline_iou"] = round(float(baseline_iou), 4)
    if area_m2 is not None:
        props["area_m2"] = round(float(area_m2), 1)
    if extra_props:
        # Carry source-feature provenance (e.g. source_gdb, SourceDate) onto
        # the output so UI layers can show "from 2017 NYC OpenData" per row.
        for k, v in extra_props.items():
            if k in props:
                continue  # never let provenance shadow computed fields
            props[k] = v
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": coordinates},
    }


# ----------------------------------------------------------------------
# Per-source-feature classification helpers
# ----------------------------------------------------------------------


def _load_baseline_source_features(work_dir: Path) -> list[dict[str, Any]] | None:
    """Load baseline_footprints.geojson features when the worker materialized
    one. Returns None when the file doesn't exist (tests, non-NYC paths)."""
    gj_path = work_dir / "baseline_footprints.geojson"
    if not gj_path.exists():
        return None
    try:
        payload = json.loads(gj_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    feats = payload.get("features")
    if not isinstance(feats, list):
        return None
    return [f for f in feats if isinstance(f, dict) and f.get("geometry")]


def _rasterize_single_geom(
    geom: dict[str, Any],
    *,
    out_shape: tuple[int, int],
    transform: Any | None,
) -> Any:
    """Rasterize one GeoJSON geometry into a boolean mask at `out_shape`.

    Returns an all-False mask when rasterization fails (bad geometry,
    no overlap with out_shape, etc).
    """
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    tr = transform if transform is not None else Affine.identity()
    try:
        mask_u8 = rasterize(
            shapes=[(geom, 1)],
            out_shape=out_shape,
            transform=tr,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        )
        return mask_u8.astype(bool)
    except Exception:
        return np.zeros(out_shape, dtype=bool)


def _extract_source_provenance(source_props: Any) -> dict[str, Any]:
    """Pick the small set of GDB-provenance fields worth forwarding."""
    if not isinstance(source_props, dict):
        return {}
    keep = ("source_gdb", "source_layer", "Source", "SourceDate", "NYSGeo_Source")
    return {k: source_props[k] for k in keep if k in source_props}


# ----------------------------------------------------------------------
# Main stage
# ----------------------------------------------------------------------


def stage_change(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    """Per-baseline-building change classification.

    Produces one GeoJSON Feature per building event:
      - unchanged   — baseline footprint well-covered by current mask (IoU >= 0.6)
      - modified    — baseline footprint partially covered (0.2 <= IoU < 0.6)
      - demolished  — baseline footprint no longer present (IoU < 0.2)
      - added       — current-year structure that doesn't overlap any baseline

    Replaces the old per-connected-component XOR-polygon output that
    emitted hundreds of edge-slivers per run.
    """
    import numpy as np

    out_path = work_dir / "change.geojson"

    imagery_mask = ctx.get("refined_mask", ctx.get("mask"))
    baseline_mask = ctx.get("baseline_footprints_mask")
    if baseline_mask is None:
        baseline_mask = ctx.get("refined_baseline_mask", ctx.get("baseline_mask"))
    if imagery_mask is None or baseline_mask is None:
        raise RuntimeError("change stage requires both imagery and baseline masks")

    transform = ctx.get("orthophoto_transform")
    crs = ctx.get("orthophoto_crs")
    if crs is None or transform is None:
        # Pixel-space only: we can still classify, but area-based filters
        # are measured in pixels rather than m².
        pixel_space_only = True
        crs_value = "pixel"
    else:
        pixel_space_only = False
        crs_value = str(crs)

    im = np.asarray(imagery_mask).astype(bool)
    base = np.asarray(baseline_mask).astype(bool)

    min_area_m2 = _min_area_m2()
    unchanged_thresh = _unchanged_iou()
    modified_thresh = _modified_iou()
    added_overlap_cap = _added_max_baseline_overlap()
    px_area_m2 = _pixel_area_m2(transform, crs_value)
    # When no CRS info is available, fall back to a pixel-count threshold
    # that's comparable across tests.
    min_area_px = (
        int(round(min_area_m2 / px_area_m2)) if px_area_m2 > 0 else _int_env("CITYLENS_CHANGE_MIN_AREA_PX", 40)
    )

    features: list[dict[str, Any]] = []
    counts = {"unchanged": 0, "modified": 0, "demolished": 0, "added": 0}

    # ------------------------------------------------------------------
    # 1. Classify each baseline footprint
    # ------------------------------------------------------------------
    # Prefer iterating the source GeoJSON features directly: one output
    # feature per GDB row (44 outputs for a Brooklyn block). If the worker
    # didn't materialize the geojson (unit tests, non-NYC paths), fall back
    # to component-labeling the rasterized baseline mask.
    source_features = _load_baseline_source_features(work_dir)
    used_per_source = False

    if source_features is not None and not pixel_space_only:
        used_per_source = True
        summary.qa["change_source"] = "per_source_feature"
        h, w = im.shape
        for src_feat in source_features:
            geom = src_feat.get("geometry")
            if not geom:
                continue
            single_mask = _rasterize_single_geom(
                geom, out_shape=(h, w), transform=transform
            )
            single_area_px = int(single_mask.sum())
            if single_area_px < 1:
                # Footprint falls outside the ortho bbox — can't classify.
                continue

            # IoU measured within the footprint's bbox (pad 10%) so a
            # single large SAM2 blob can't swallow every neighbor.
            ys, xs = np.where(single_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            pad_y = max(1, (y1 - y0) // 10)
            pad_x = max(1, (x1 - x0) // 10)
            y0 = max(0, y0 - pad_y)
            x0 = max(0, x0 - pad_x)
            y1 = min(h, y1 + pad_y)
            x1 = min(w, x1 + pad_x)
            fp_roi = single_mask[y0:y1, x0:x1]
            im_roi = im[y0:y1, x0:x1]
            inter = int(np.logical_and(fp_roi, im_roi).sum())
            union = int(np.logical_or(fp_roi, im_roi).sum())
            iou = float(inter) / float(union) if union > 0 else 0.0

            if iou >= unchanged_thresh:
                change_type = "unchanged"
            elif iou >= modified_thresh:
                change_type = "modified"
            else:
                change_type = "demolished"
            counts[change_type] += 1

            area_m2_val = float(single_area_px) * px_area_m2 if px_area_m2 > 0 else None
            provenance = _extract_source_provenance(src_feat.get("properties"))

            # Emit the footprint geometry verbatim from the source geojson
            # (preserves shared edges of row houses instead of bleeding into
            # the neighbor via rasterize+shapes).
            src_type = str(geom.get("type", "")).strip()
            src_coords = geom.get("coordinates")
            if src_type == "Polygon" and src_coords:
                poly_list = [src_coords]
            elif src_type == "MultiPolygon" and src_coords:
                poly_list = list(src_coords)
            else:
                poly_list = []

            for coordinates in poly_list:
                features.append(
                    _feature(
                        change_type=change_type,
                        coordinates=coordinates,
                        area_m2=area_m2_val,
                        baseline_iou=iou,
                        imagery_year=request.imagery_year,
                        baseline_year=request.baseline_year,
                        crs_value=crs_value,
                        extra_props=provenance,
                    )
                )
    else:
        summary.qa["change_source"] = "component_labeled"
        baseline_labels, baseline_n = _label_components(base)
        for comp_id in range(1, baseline_n + 1):
            comp = baseline_labels == comp_id
            comp_area_px = int(comp.sum())
            if comp_area_px < 1:
                continue

            ys, xs = np.where(comp)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            pad_y = max(1, (y1 - y0) // 10)
            pad_x = max(1, (x1 - x0) // 10)
            y0 = max(0, y0 - pad_y)
            x0 = max(0, x0 - pad_x)
            y1 = min(comp.shape[0], y1 + pad_y)
            x1 = min(comp.shape[1], x1 + pad_x)

            comp_roi = comp[y0:y1, x0:x1]
            im_roi = im[y0:y1, x0:x1]
            inter = int(np.logical_and(comp_roi, im_roi).sum())
            union = int(np.logical_or(comp_roi, im_roi).sum())
            iou = float(inter) / float(union) if union > 0 else 0.0

            if iou >= unchanged_thresh:
                change_type = "unchanged"
            elif iou >= modified_thresh:
                change_type = "modified"
            else:
                change_type = "demolished"
            counts[change_type] += 1

            area_m2_val = float(comp_area_px) * px_area_m2 if px_area_m2 > 0 else None
            polys = _polygon_coords_from_pixel_mask(comp, transform=transform)
            for coordinates in polys:
                features.append(
                    _feature(
                        change_type=change_type,
                        coordinates=coordinates,
                        area_m2=area_m2_val,
                        baseline_iou=iou,
                        imagery_year=request.imagery_year,
                        baseline_year=request.baseline_year,
                        crs_value=crs_value,
                    )
                )

    # ------------------------------------------------------------------
    # 2. Find "added" buildings — components in imagery that don't overlap
    #    any baseline footprint.
    # ------------------------------------------------------------------
    # LiDAR height gate. Pulled from ctx if refine managed to sample a
    # dense grid; `None` means "no LiDAR available, don't reject on height".
    lidar_heights = ctx.get("lidar_heights")
    lidar_ground_z = ctx.get("lidar_ground_z")
    added_min_height_m = _added_min_height_m()
    added_height_pctl = _added_height_percentile()
    added_reject_reasons = {"too_small": 0, "baseline_overlap": 0, "too_short": 0}

    added_pixels = np.logical_and(im, np.logical_not(base))
    added_labels, added_n = _label_components(added_pixels)
    for comp_id in range(1, added_n + 1):
        comp = added_labels == comp_id
        comp_area_px = int(comp.sum())
        if comp_area_px < min_area_px:
            added_reject_reasons["too_small"] += 1
            continue

        # Reject slivers along existing baseline buildings.
        overlap_touching = int(np.logical_and(comp, _one_step_dilate(base)).sum())
        overlap_fraction = float(overlap_touching) / float(comp_area_px) if comp_area_px else 1.0
        if overlap_fraction > added_overlap_cap:
            added_reject_reasons["baseline_overlap"] += 1
            continue

        # Reject things SAM2 thinks are buildings but which LiDAR says are
        # short — trees, hedges, vehicles, pavement patterns. Only gated
        # when we have LiDAR + a ground-plane estimate.
        if lidar_heights is not None and lidar_ground_z is not None:
            cell_heights = np.asarray(lidar_heights)[comp]
            finite = cell_heights[np.isfinite(cell_heights)]
            if finite.size == 0:
                # No LiDAR coverage for this component. Err on the side of
                # rejecting: in a real NYC scene the LiDAR tile usually
                # covers real buildings. If it doesn't cover this blob,
                # the blob is probably outside the tile's footprint too.
                added_reject_reasons["too_short"] += 1
                continue
            pctl_z = float(np.percentile(finite, added_height_pctl))
            # sample_lidar_heights returns meters already, no unit conversion.
            height_above_ground_m = pctl_z - float(lidar_ground_z)
            if height_above_ground_m < added_min_height_m:
                added_reject_reasons["too_short"] += 1
                continue

        counts["added"] += 1
        area_m2_val = float(comp_area_px) * px_area_m2 if px_area_m2 > 0 else None
        polys = _polygon_coords_from_pixel_mask(comp, transform=transform)
        for coordinates in polys:
            features.append(
                _feature(
                    change_type="added",
                    coordinates=coordinates,
                    area_m2=area_m2_val,
                    baseline_iou=None,
                    imagery_year=request.imagery_year,
                    baseline_year=request.baseline_year,
                    crs_value=crs_value,
                )
            )

    feature_collection = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(feature_collection, indent=2))

    # Surface breakdown on summary.qa so operators can eyeball noise levels
    # without parsing the geojson.
    summary.qa["change_counts"] = dict(counts)
    # Why the "added" gate rejected candidates — diagnostic signal for
    # tuning. Empty buckets are fine. total = rejected; accepted is
    # change_counts["added"].
    summary.qa["added_rejected"] = dict(added_reject_reasons)

    change_mask = np.logical_or(
        np.logical_and(im, np.logical_not(base)),
        np.logical_and(base, np.logical_not(im)),
    ).astype(np.uint8)
    return {**ctx, "change_path": out_path, "change_mask": change_mask}


def _one_step_dilate(mask):
    """Tiny 3x3 dilation: used to detect 'component touches baseline' for
    the added-sliver filter. Pure numpy; avoids scipy/skimage."""
    import numpy as np

    m = np.asarray(mask).astype(bool)
    if not m.any():
        return m
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy : dy + m.shape[0], dx : dx + m.shape[1]]
    return out
