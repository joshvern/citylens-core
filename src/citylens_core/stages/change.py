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


def _added_baseline_dilate_px() -> int:
    """Pixel-radius dilation of the baseline mask used by the courtyard
    filter on 'added' candidates.

    The 1-pixel `_one_step_dilate` overlap filter only catches candidates
    that physically TOUCH a baseline footprint — but courtyards / lightwells
    typically sit in a 2-10 pixel gap between buildings (the GDB rasterizer
    snaps to whole-pixel building edges and there's a small road / sidewalk
    moat). This wider dilation lets us reject any candidate whose centroid
    falls inside the dilated baseline mask, catching courtyards entirely
    surrounded by buildings without rejecting genuine new construction
    further away.

    Default 8 px ≈ 4 m at 0.5 m/px — wide enough to bridge most rasterized
    courtyard gaps, narrow enough that a real new building one parcel over
    is still flagged.
    """
    return _int_env("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", 8)


def _demolished_max_height_m() -> float:
    """LiDAR-validated demolished gate.

    The current rule labels a baseline footprint as "demolished" whenever
    SAM2 IoU < modified_iou (0.2 by default). But SAM2 sometimes misses
    buildings entirely — dark roofs on shadowed imagery come out IoU≈0 and
    get wrongly labeled demolished. If LiDAR shows a real structure
    standing inside the baseline footprint (75th-percentile height above
    ground exceeds this threshold), downgrade the classification from
    "demolished" → "modified" — the SAM2 mask is unreliable but LiDAR
    confirms a building is still there.

    Default 3.0 m: taller than a parked truck or hedge, shorter than every
    real one-story building.
    """
    return _float_env("CITYLENS_CHANGE_DEMOLISHED_MAX_HEIGHT_M", 3.0)


def _demolished_height_percentile() -> float:
    """Percentile of LiDAR z-values within a baseline footprint used by the
    demolished-rescue gate. Same rationale as `_added_height_percentile` —
    the 75th percentile is robust to a single tall tree leaning over an
    otherwise-empty parcel."""
    return _float_env("CITYLENS_CHANGE_DEMOLISHED_HEIGHT_PERCENTILE", 75.0)


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
    height_m: float | None = None,
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
    if height_m is not None:
        # 95th-pct LiDAR height above ground inside this footprint, in
        # meters. Lets the UI show "32 m, ~10 stories" without the user
        # parsing the mesh PLY. None when no LiDAR coverage was available.
        props["height_m"] = round(float(height_m), 1)
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

    # LiDAR-validated demolished rescue. Pulled up here so both the
    # per-source-feature path and the legacy component-labeled fallback can
    # use it. `None` ⇒ no LiDAR available, rescue disabled (legacy
    # behavior).
    lidar_heights = ctx.get("lidar_heights")
    lidar_ground_z = ctx.get("lidar_ground_z")
    demolished_max_h = _demolished_max_height_m()
    demolished_pctl = _demolished_height_percentile()
    summary.qa.setdefault("demolished_downgraded_to_modified", 0)

    def _classify_with_lidar_rescue(iou: float, footprint_mask) -> str:
        """Apply IoU bands, then downgrade demolished→modified if LiDAR
        shows a real structure standing inside the baseline footprint."""
        if iou >= unchanged_thresh:
            return "unchanged"
        if iou >= modified_thresh:
            return "modified"
        # IoU < modified_thresh — would normally be 'demolished'. Try LiDAR
        # rescue: if the 75th-pct height above ground is more than
        # `demolished_max_h` meters, SAM2 likely missed the building (dark
        # roof, shadow), so downgrade to 'modified' instead of demolished.
        if lidar_heights is not None and lidar_ground_z is not None:
            h_above = _lidar_height_at_mask(
                mask=footprint_mask,
                lidar_heights=lidar_heights,
                lidar_ground_z=float(lidar_ground_z),
                percentile=demolished_pctl,
            )
            if h_above is not None and h_above > demolished_max_h:
                summary.qa["demolished_downgraded_to_modified"] += 1
                return "modified"
        return "demolished"

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

            change_type = _classify_with_lidar_rescue(iou, single_mask)
            counts[change_type] += 1

            area_m2_val = float(single_area_px) * px_area_m2 if px_area_m2 > 0 else None
            provenance = _extract_source_provenance(src_feat.get("properties"))
            # Sample 95th-pct LiDAR height inside the footprint so each output
            # feature carries its own building height. Reused by UIs and by
            # the LOD1 mesh stage. None when LiDAR isn't available.
            feature_height_m: float | None = None
            if lidar_heights is not None and lidar_ground_z is not None:
                feature_height_m = _lidar_height_at_mask(
                    mask=single_mask,
                    lidar_heights=lidar_heights,
                    lidar_ground_z=float(lidar_ground_z),
                    percentile=95.0,
                )

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
                        height_m=feature_height_m,
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

            change_type = _classify_with_lidar_rescue(iou, comp)
            counts[change_type] += 1

            area_m2_val = float(comp_area_px) * px_area_m2 if px_area_m2 > 0 else None
            comp_height_m: float | None = None
            if lidar_heights is not None and lidar_ground_z is not None:
                comp_height_m = _lidar_height_at_mask(
                    mask=comp,
                    lidar_heights=lidar_heights,
                    lidar_ground_z=float(lidar_ground_z),
                    percentile=95.0,
                )
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
                        height_m=comp_height_m,
                    )
                )

    # ------------------------------------------------------------------
    # 2. Find "added" buildings — components in imagery that don't overlap
    #    any baseline footprint.
    # ------------------------------------------------------------------
    added_min_height_m = _added_min_height_m()
    added_height_pctl = _added_height_percentile()
    added_baseline_dilate_px = _added_baseline_dilate_px()
    added_reject_reasons = {
        "too_small": 0,
        "baseline_overlap": 0,
        "centroid_near_baseline": 0,
        "too_short": 0,
    }

    # Pre-compute the wide baseline dilation used by the courtyard /
    # centroid-near-baseline filter. Built once outside the per-component
    # loop because each step is O(H*W) and we'd otherwise pay it `added_n`
    # times.
    baseline_dilated_wide = (
        _n_step_dilate(base, added_baseline_dilate_px)
        if added_baseline_dilate_px > 0
        else None
    )

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

        # Reject courtyards / lightwells: a candidate whose centroid lies
        # inside a wider dilation of the baseline mask is almost certainly
        # an interior gap between buildings (rasterized GDB footprints
        # snap to whole-pixel edges and there's a small road/sidewalk gap,
        # so the 1-px overlap filter above doesn't catch them).
        if baseline_dilated_wide is not None:
            ys_c, xs_c = np.where(comp)
            cy = int(round(float(ys_c.mean())))
            cx = int(round(float(xs_c.mean())))
            h_, w_ = baseline_dilated_wide.shape
            cy = max(0, min(h_ - 1, cy))
            cx = max(0, min(w_ - 1, cx))
            if bool(baseline_dilated_wide[cy, cx]):
                added_reject_reasons["centroid_near_baseline"] += 1
                continue

        # Reject things SAM2 thinks are buildings but which LiDAR says are
        # short — trees, hedges, vehicles, pavement patterns. Only gated
        # when we have LiDAR + a ground-plane estimate.
        added_height_above_ground_m: float | None = None
        if lidar_heights is not None and lidar_ground_z is not None:
            added_height_above_ground_m = _lidar_height_at_mask(
                mask=comp,
                lidar_heights=lidar_heights,
                lidar_ground_z=float(lidar_ground_z),
                percentile=added_height_pctl,
            )
            if added_height_above_ground_m is None:
                # No LiDAR coverage for this component. Err on the side of
                # rejecting: in a real NYC scene the LiDAR tile usually
                # covers real buildings. If it doesn't cover this blob,
                # the blob is probably outside the tile's footprint too.
                added_reject_reasons["too_short"] += 1
                continue
            if added_height_above_ground_m < added_min_height_m:
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
                    height_m=added_height_above_ground_m,
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


def _n_step_dilate(mask, steps: int):
    """Iterated 3x3 dilation for `steps` rounds. Equivalent to a binary
    dilation by an `(2*steps+1) x (2*steps+1)` square kernel.

    Pure numpy (no scipy/skimage) — same pattern as `_one_step_dilate`,
    just looped. Used by the courtyard filter on 'added' candidates: an
    8-step dilation of the baseline mask widens each footprint by ~4 m at
    0.5 m/px, closing the typical road/sidewalk gap so candidates whose
    centroid lies inside the dilated mask (i.e. surrounded by buildings)
    can be rejected.
    """
    import numpy as np

    m = np.asarray(mask).astype(bool)
    if steps <= 0 or not m.any():
        return m
    out = m
    for _ in range(int(steps)):
        out = _one_step_dilate(out)
    return out


def _lidar_height_at_mask(
    *, mask, lidar_heights, lidar_ground_z: float, percentile: float
) -> float | None:
    """Return percentile-of-LiDAR height above ground inside `mask`, or
    None when LiDAR has no finite samples covering the mask region.

    Centralizes the height-sampling logic shared by the 'added' height
    gate and the 'demolished' rescue check.
    """
    import numpy as np

    cell_heights = np.asarray(lidar_heights)[mask]
    finite = cell_heights[np.isfinite(cell_heights)]
    if finite.size == 0:
        return None
    pctl_z = float(np.percentile(finite, percentile))
    return pctl_z - float(lidar_ground_z)
