from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary
from ._polygon_smoothing import (
    estimate_pixel_tolerance_in_world_units,
    simplify_polygon_coords,
)
from ._registration import apply_shift, estimate_alignment
from ._surface_change import (
    is_surface_changed,
    load_surface_images,
    surface_delta_e,
)

logger = logging.getLogger(__name__)


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


def _bool_env(name: str, default: bool) -> bool:
    """Truthy/falsy parse for env vars. Accepts 1/0, true/false, yes/no, on/off
    (case-insensitive). Anything else falls back to the provided default — we
    never want a typo in a tunable to silently flip behavior."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _min_area_m2() -> float:
    """Drop change features smaller than this area in square meters.

    Default 100 m² (roughly one small garage/shed). Kills the sliver-noise
    from imperfect mask alignment at building edges.
    """
    return _float_env("CITYLENS_CHANGE_MIN_AREA_M2", 100.0)


def _unchanged_iou() -> float:
    # Default ceiling on the unchanged threshold. The actual threshold used
    # at runtime is min(this, median_baseline_iou - 0.10), so a clean tile
    # whose median IoU is high stays at this default while a noisy tile
    # whose median IoU is low gets a more lenient threshold (still floored
    # by _unchanged_iou_floor below).
    #
    # Calibration history:
    # - 0.6 (initial guess) flagged 30/43 Brooklyn brownstones as 'modified'.
    # - 0.5 (after first calibration) hit 44% modified on the wider 169-bldg
    #   AOI.
    # - 0.4 (current default) works for clean tiles like Brooklyn but still
    #   flags 60%+ on dense Manhattan mixed-use (Cooper Square, median IoU
    #   ~0.34). The adaptive logic in stage_change handles those by lowering
    #   to median - 0.10.
    return _float_env("CITYLENS_CHANGE_UNCHANGED_IOU", 0.4)


def _unchanged_iou_floor() -> float:
    """Adaptive-threshold floor: never use an unchanged_iou below this even
    when the median IoU on a tile is very low. Prevents pathological
    tiles where SAM2 is essentially unusable from collapsing all buildings
    into 'unchanged'."""
    return _float_env("CITYLENS_CHANGE_UNCHANGED_IOU_FLOOR", 0.25)


def _adaptive_threshold_min_samples() -> int:
    """Below this many baseline IoU samples the adaptive logic doesn't kick
    in — small/synthetic tiles use the configured `unchanged_iou` directly.
    """
    return _int_env("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", 20)


def _modified_iou() -> float:
    return _float_env("CITYLENS_CHANGE_MODIFIED_IOU", 0.2)


def _modified_borderline_margin() -> float:
    """IoU width below the (post-adaptive) unchanged threshold inside which a
    "modified" classification is treated as a borderline call.

    Built to address the dominant production failure mode observed across
    Manhattan / Williamsburg demos (Cooper Square, Hudson Yards, LIC Borden,
    Bedford-Williamsburg): the per-footprint IoU distribution clusters just
    below the unchanged threshold, so 10–25 % of buildings get flagged as
    "modified" between 2017 and 2024 even though most of those neighborhoods
    didn't actually change. The IoU drop in those cases is segmentation
    noise and sub-pixel registration drift, NOT a real building modification.

    Set to 0 to disable the borderline-reclassification pass entirely (the
    pass otherwise demotes borderline modifieds to unchanged unless the
    surface-change Δ-E signal independently confirms a visual change). The
    historical pre-margin behavior is recovered with margin=0.

    Default 0.05 (5 % IoU): roughly the magnitude of the per-footprint IoU
    drop attributable to a 1 px registration error on a 30–60 m² rooftop at
    0.5 m/px. Anything within this band of the unchanged threshold is in
    "could go either way" territory and the pass treats it as unchanged
    until proven otherwise.
    """
    return _float_env("CITYLENS_CHANGE_MODIFIED_BORDERLINE_MARGIN", 0.05)


def _modified_borderline_require_surface_evidence() -> bool:
    """Policy switch for the borderline-modified reclassification pass.

    True (default): a borderline modified is reclassified to unchanged
        UNLESS surface_changed=True confirms a real visual change. This is
        the noise-reducing direction — what we want in production today
        where the surface-change signal is dormant (no RGB baseline) and
        any "modified" call near the unchanged threshold is presumptively
        segmentation noise.

    False: only reclassify when surface evidence is available AND it says
        no change. This is the historically more permissive direction
        (innocent until proven guilty); useful as an ablation toggle when
        a real RGB 2017 baseline gets wired up and surface_changed becomes
        a reliable signal.
    """
    return _bool_env(
        "CITYLENS_CHANGE_MODIFIED_BORDERLINE_REQUIRE_SURFACE", True
    )


def _registration_max_shift_px() -> int:
    """Cap on the per-tile (dy, dx) translation phase-correlation will
    consider. Default 4 px is conservative — registration error between
    NYS Orthos baseline and current acquisitions is typically < 2 px."""
    return _int_env("CITYLENS_CHANGE_REGISTRATION_MAX_SHIFT_PX", 4)


def _registration_min_confidence() -> float:
    """Phase-correlation confidence floor. Below this, the estimated
    shift is treated as noise and not applied."""
    return _float_env("CITYLENS_CHANGE_REGISTRATION_MIN_CONFIDENCE", 0.15)


def _registration_min_iou_gain() -> float:
    """Minimum IoU improvement required to actually apply a registration
    shift. A trivial gain (< 0.01) isn't worth the risk."""
    return _float_env("CITYLENS_CHANGE_REGISTRATION_MIN_IOU_GAIN", 0.01)


def _surface_delta_e_threshold() -> float:
    """CIE76 Lab Delta-E above which a footprint with high IoU is
    flagged as surface_changed. 20 ≈ "clearly different color/material"
    on the perceptual scale (0–100). Below this, seasonal lighting
    drift dominates."""
    return _float_env("CITYLENS_CHANGE_SURFACE_DELTA_E", 20.0)


def _polygon_simplification_pixel_tolerance() -> float:
    """Douglas-Peucker tolerance in PIXELS. Multiplied by world-unit
    pixel size when emitting georeferenced polygons. 0.5 px collapses
    saw-tooth rasterization without losing real corners. Set to 0 to
    disable simplification entirely."""
    return _float_env("CITYLENS_CHANGE_POLYGON_SIMPLIFY_PIXELS", 0.5)


def _candidate_added_confidence() -> float:
    """Confidence value attached to "added" components emitted as
    candidate_added (rejected by the LiDAR-coverage gate but otherwise
    valid). Frontend can render these faded."""
    return _float_env("CITYLENS_CHANGE_CANDIDATE_ADDED_CONFIDENCE", 0.3)


# ----------------------------------------------------------------------
# Confidence scoring (#3): every emitted feature gets a 0..1 confidence
# derived from how far it sits from each classifier's threshold band.
# ----------------------------------------------------------------------


def _classification_confidence(
    *,
    change_type: str,
    iou: float | None,
    unchanged_thresh: float,
    modified_thresh: float,
    lidar_rescued: bool = False,
    surface_changed: bool = False,
) -> float:
    """Derive a 0..1 confidence score for a classified feature.

    The signal comes from how far the feature's IoU sits from the band
    edges. Features in the middle of a band get high confidence;
    features near a boundary get low confidence. Special cases:
    - LiDAR-rescued demolished->modified gets capped confidence (we
      believe the building exists but SAM2 couldn't see it).
    - Surface-changed unchanged gets full confidence (independent
      evidence beyond IoU).
    """
    if iou is None:
        # "added" path — caller should pass a context-specific
        # confidence; the default base is 0.7 which the candidate_added
        # path overrides via _candidate_added_confidence().
        return 0.7

    if change_type == "unchanged":
        # Distance above unchanged threshold, normalized to [0, 1] over
        # the [thresh, 1.0] interval, then mapped to [0.6, 1.0] so even
        # boundary-hugging unchanged features get a meaningful confidence.
        span = max(1e-6, 1.0 - unchanged_thresh)
        normalized = max(0.0, min(1.0, (iou - unchanged_thresh) / span))
        conf = 0.6 + 0.4 * normalized
        if surface_changed:
            # Independent evidence reinforces the unchanged-shape call.
            conf = max(conf, 0.85)
        return float(conf)

    if change_type == "modified":
        # Distance from BOTH band edges. Highest confidence in the middle.
        center = 0.5 * (modified_thresh + unchanged_thresh)
        half_band = max(1e-6, 0.5 * (unchanged_thresh - modified_thresh))
        proximity = max(0.0, 1.0 - abs(iou - center) / half_band)
        conf = 0.5 + 0.4 * proximity
        return float(conf)

    if change_type == "demolished":
        if lidar_rescued:
            # SAM2 missed the roof but LiDAR shows a building. Result was
            # downgraded to modified; this branch shouldn't fire, but if
            # it does the LiDAR evidence is moderate.
            return 0.5
        # Pure demolished: lower IoU = higher confidence the building
        # is gone. iou=0 → ~1.0, iou near modified_thresh → ~0.5.
        span = max(1e-6, modified_thresh)
        normalized = max(0.0, min(1.0, (modified_thresh - iou) / span))
        return float(0.5 + 0.5 * normalized)

    return 0.7


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

    # Compute the simplification tolerance once per stage call. Half a
    # pixel collapses saw-tooth rasterization while preserving real
    # geometry. Set CITYLENS_CHANGE_POLYGON_SIMPLIFY_PIXELS=0 to skip.
    pixel_tol = _polygon_simplification_pixel_tolerance()
    world_tol = (
        estimate_pixel_tolerance_in_world_units(transform, pixel_tolerance=pixel_tol)
        if pixel_tol > 0
        else 0.0
    )

    for geom, value in shapes(m_u8, mask=m, transform=tr):
        if int(value) != 1:
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        rings = [list(ring) for ring in coords]
        if world_tol > 0:
            rings = simplify_polygon_coords(rings, tolerance=world_tol)
        polys.append(rings)
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
    confidence: float | None = None,
    surface_delta_e_value: float | None = None,
    surface_changed: bool | None = None,
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
    if confidence is not None:
        # 0..1 — how confident the classifier is in this feature's
        # change_type. Frontend can render low-confidence features
        # faded.
        props["confidence"] = round(float(max(0.0, min(1.0, confidence))), 3)
    if surface_delta_e_value is not None:
        # Median CIE76 Lab Delta-E inside the footprint, between
        # baseline and current imagery. Useful even when surface_changed
        # is False as a calibration signal.
        props["surface_delta_e"] = round(float(surface_delta_e_value), 2)
    if surface_changed is not None:
        props["surface_changed"] = bool(surface_changed)
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

    # ------------------------------------------------------------------
    # Sub-pixel registration (#1). Estimate one global (dy, dx) shift
    # that aligns the baseline mask to the current mask. Recovers the
    # per-footprint IoU lost to acquisition-year registration error
    # (typically 0.5–2 px between NYS Orthos baseline and current).
    # If the phase-correlation peak isn't confident enough or the IoU
    # gain is trivial, the shift is NOT applied — defensive default.
    # ------------------------------------------------------------------
    registration = estimate_alignment(
        base,
        im,
        max_shift_px=_registration_max_shift_px(),
        min_confidence=_registration_min_confidence(),
        min_iou_gain=_registration_min_iou_gain(),
    )
    summary.qa["registration"] = {
        "dy": registration.dy,
        "dx": registration.dx,
        "confidence": round(registration.confidence, 3),
        "iou_before": round(registration.iou_before, 4),
        "iou_after": round(registration.iou_after, 4),
        "applied": bool(registration.accepted),
    }
    if registration.accepted:
        base = apply_shift(base, dy=registration.dy, dx=registration.dx)
        # Keep the rasterized-footprints map in sync if the per-source
        # path later uses it (it doesn't currently — it rasterizes from
        # GeoJSON directly — but we set it anyway for any downstream
        # ctx readers).
        if ctx.get("baseline_footprints_mask") is baseline_mask:
            ctx["baseline_footprints_mask"] = base

    # ------------------------------------------------------------------
    # Surface-change detection (#2). Re-load the orthophoto + baseline
    # imagery so we can measure perceptual color delta inside high-IoU
    # footprints. Lets us flag re-roofing / repainting that shape-only
    # IoU classifies as "unchanged".
    # ------------------------------------------------------------------
    surface_images = load_surface_images(
        orthophoto_path=ctx.get("orthophoto_path"),
        baseline_path=ctx.get("baseline_path"),
        expected_shape=im.shape,
    )
    surface_threshold = _surface_delta_e_threshold()
    summary.qa["surface_change_available"] = bool(surface_images is not None)

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

    if source_features is not None and not pixel_space_only:
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

            # Surface-change check (#2). Computed for `unchanged` features
            # (catches re-roofing / repainting that shape-only IoU misses)
            # AND for `modified` features (so the borderline-modified
            # reclassification pass below can use surface_changed as a
            # CONFIRMATION signal that a near-threshold IoU drop reflects a
            # real visual change rather than segmentation noise).
            de_value: float | None = None
            surface_flag: bool | None = None
            if (
                change_type in ("unchanged", "modified")
                and surface_images is not None
            ):
                de_value = surface_delta_e(
                    images=surface_images,
                    footprint_mask=single_mask,
                    erode_px=1,
                )
                surface_flag = is_surface_changed(de_value, threshold=surface_threshold)

            confidence = _classification_confidence(
                change_type=change_type,
                iou=iou,
                unchanged_thresh=unchanged_thresh,
                modified_thresh=modified_thresh,
                surface_changed=bool(surface_flag),
            )

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
                        confidence=confidence,
                        surface_delta_e_value=de_value,
                        surface_changed=surface_flag,
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
            # Surface-change check (#2) — same as per-source path, fires
            # on both unchanged AND modified components so the borderline-
            # modified reclassification pass below has Δ-E evidence to
            # work with. Pixel-space-only fallback tiles probably won't
            # have surface_images either.
            de_value: float | None = None
            surface_flag: bool | None = None
            if (
                change_type in ("unchanged", "modified")
                and surface_images is not None
            ):
                de_value = surface_delta_e(
                    images=surface_images,
                    footprint_mask=comp,
                    erode_px=1,
                )
                surface_flag = is_surface_changed(de_value, threshold=surface_threshold)

            confidence = _classification_confidence(
                change_type=change_type,
                iou=iou,
                unchanged_thresh=unchanged_thresh,
                modified_thresh=modified_thresh,
                surface_changed=bool(surface_flag),
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
                        confidence=confidence,
                        surface_delta_e_value=de_value,
                        surface_changed=surface_flag,
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
        "no_lidar_coverage_emitted_as_candidate": 0,
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
        emit_as_candidate = False  # #5: True iff LiDAR has no coverage for this comp.
        if lidar_heights is not None and lidar_ground_z is not None:
            added_height_above_ground_m = _lidar_height_at_mask(
                mask=comp,
                lidar_heights=lidar_heights,
                lidar_ground_z=float(lidar_ground_z),
                percentile=added_height_pctl,
            )
            if added_height_above_ground_m is None:
                # No LiDAR coverage for this component. Don't fully drop
                # it — the candidate may be a real building outside the
                # LiDAR tile's footprint. Emit with change_type =
                # "candidate_added" and a low confidence, so frontends
                # can render it faded ("possibly added, no LiDAR
                # confirmation").
                added_reject_reasons["no_lidar_coverage_emitted_as_candidate"] += 1
                emit_as_candidate = True
            elif added_height_above_ground_m < added_min_height_m:
                added_reject_reasons["too_short"] += 1
                continue

        if emit_as_candidate:
            change_type_emitted = "candidate_added"
            counts.setdefault("candidate_added", 0)
            counts["candidate_added"] += 1
            confidence_emitted = _candidate_added_confidence()
        else:
            change_type_emitted = "added"
            counts["added"] += 1
            confidence_emitted = _classification_confidence(
                change_type="added",
                iou=None,
                unchanged_thresh=unchanged_thresh,
                modified_thresh=modified_thresh,
            )

        area_m2_val = float(comp_area_px) * px_area_m2 if px_area_m2 > 0 else None
        polys = _polygon_coords_from_pixel_mask(comp, transform=transform)
        for coordinates in polys:
            features.append(
                _feature(
                    change_type=change_type_emitted,
                    coordinates=coordinates,
                    area_m2=area_m2_val,
                    baseline_iou=None,
                    imagery_year=request.imagery_year,
                    baseline_year=request.baseline_year,
                    crs_value=crs_value,
                    height_m=added_height_above_ground_m,
                    confidence=confidence_emitted,
                )
            )

    # ------------------------------------------------------------------
    # 3. Adaptive unchanged_iou threshold
    # ------------------------------------------------------------------
    # SAM2's segmentation quality varies a lot between tiles. On clean
    # Brooklyn brownstones the per-baseline IoU clusters around 0.6, so a
    # 0.4 threshold cleanly separates unchanged from modified. On dense
    # mixed-use blocks (East Village, Williamsburg) SAM2's median IoU
    # drops to ~0.3, and 0.4 then flags 60-70% of buildings as "modified"
    # on stable blocks — the user perceives this as the model
    # hallucinating change. Lower the threshold per-tile when the
    # distribution shifts down. Cap at the configured default so clean
    # tiles don't regress.
    iou_samples = [
        f["properties"]["baseline_iou"]
        for f in features
        if isinstance(f.get("properties"), dict)
        and f["properties"].get("baseline_iou") is not None
    ]
    unchanged_thresh_used = unchanged_thresh
    # Need a real sample size to trust the median — the adaptive logic is
    # only meaningful on a real-world tile with many baselines, never on
    # tests / synthetic fixtures where we'd otherwise classify the only
    # input feature against its own IoU.
    if len(iou_samples) >= _adaptive_threshold_min_samples():
        median_iou = float(np.median(iou_samples))
        summary.qa["median_baseline_iou"] = round(median_iou, 4)
        adaptive = max(_unchanged_iou_floor(), min(unchanged_thresh, median_iou - 0.10))
        if adaptive + 1e-9 < unchanged_thresh:
            # Reclassify only the unchanged↔modified swing band on baseline-
            # derived features. Don't touch demolished/added — those bands
            # are LiDAR-validated already.
            recls = {"unchanged_to_modified": 0, "modified_to_unchanged": 0}
            for f in features:
                props = f["properties"]
                iou = props.get("baseline_iou")
                if iou is None:
                    continue
                ct = props.get("change_type")
                if ct in ("added", "demolished"):
                    continue
                if ct == "modified" and iou >= adaptive:
                    counts["modified"] -= 1
                    counts["unchanged"] += 1
                    props["change_type"] = "unchanged"
                    recls["modified_to_unchanged"] += 1
                elif ct == "unchanged" and iou < adaptive and iou >= modified_thresh:
                    counts["unchanged"] -= 1
                    counts["modified"] += 1
                    props["change_type"] = "modified"
                    recls["unchanged_to_modified"] += 1
            summary.qa["adaptive_threshold_reclassifications"] = dict(recls)
            unchanged_thresh_used = adaptive
            summary.qa["change_counts"] = dict(counts)
    summary.qa["unchanged_iou_used"] = round(unchanged_thresh_used, 4)

    # ------------------------------------------------------------------
    # 4. Borderline-modified reclassification (two-stage classifier).
    # ------------------------------------------------------------------
    # PROBLEM (observed across Cooper Square, Hudson Yards, LIC Borden,
    # Bedford-Williamsburg demos): the per-footprint IoU distribution
    # bunches up just below the (post-adaptive) unchanged threshold —
    # e.g. on Hudson Yards every one of the 11 "modified" features had
    # IoU in [0.26, 0.37] with the threshold at 0.37. These are not real
    # building modifications between 2017 and 2024; they're segmentation
    # noise + sub-pixel registration drift parading as change.
    #
    # FIX: a feature classified as "modified" whose IoU sits within
    # `borderline_margin` of the unchanged threshold is treated as a
    # borderline call. It KEEPS its modified label only when surface_changed
    # independently confirms a real visual change (Δ-E above threshold);
    # otherwise it gets reclassified to "unchanged" with a low confidence
    # and a `borderline_reclassified=True` provenance flag so the UI can
    # render it differently / so operators can audit the decision.
    #
    # Like the adaptive-threshold pass above this only fires on tiles
    # with enough baseline samples to trust the IoU statistics — single-
    # feature unit tests retain their pre-margin behavior.
    #
    # Demolished and added are NOT touched — they're LiDAR-validated.
    # Demolished→modified rescues (IoU < modified_thresh, downgraded) are
    # also untouched: their IoU sits BELOW the band so the borderline-lo
    # check excludes them by construction.
    border_margin = _modified_borderline_margin()
    border_strict = _modified_borderline_require_surface_evidence()
    border_recls = {
        "to_unchanged": 0,
        "kept_by_surface_change": 0,
        "kept_lenient_no_surface_signal": 0,
    }
    if (
        border_margin > 0
        and len(iou_samples) >= _adaptive_threshold_min_samples()
        and unchanged_thresh_used > modified_thresh
    ):
        # Lower bound of the borderline band. Clamp so we never reach
        # below modified_thresh (that's the demolished-rescue territory
        # and shouldn't be touched by this pass).
        border_lo = max(modified_thresh, unchanged_thresh_used - border_margin)
        if border_lo + 1e-9 < unchanged_thresh_used:
            for f in features:
                props = f["properties"]
                if props.get("change_type") != "modified":
                    continue
                iou = props.get("baseline_iou")
                if iou is None or iou < border_lo:
                    continue  # outside the borderline band

                # surface_changed True = independent visual evidence of a
                # real modification. Keep this feature as modified.
                if props.get("surface_changed") is True:
                    border_recls["kept_by_surface_change"] += 1
                    continue

                # No surface confirmation. Decide based on the policy:
                #  - strict (default): reclassify to unchanged. The "drop
                #    in IoU" is presumed to be noise.
                #  - lenient: only reclassify when surface evidence was
                #    AVAILABLE and explicitly said "no change" (i.e.,
                #    surface_changed is False, not None). When the signal
                #    is missing entirely (None), keep as modified.
                surface_was_available = props.get("surface_changed") is False
                if not border_strict and not surface_was_available:
                    border_recls["kept_lenient_no_surface_signal"] += 1
                    continue

                counts["modified"] -= 1
                counts["unchanged"] += 1
                props["change_type"] = "unchanged"
                # Penalize the confidence — this is a noisy borderline
                # call, NOT a clean unchanged. Caps at 0.55 (just above
                # the 0.5 lower bound of the modified-band confidence
                # formula) so the frontend can render these faded.
                if props.get("confidence") is not None:
                    props["confidence"] = round(
                        min(float(props["confidence"]), 0.55), 3
                    )
                else:
                    props["confidence"] = 0.55
                # Provenance flag: this feature didn't go through the
                # normal "unchanged" gate; downstream consumers can
                # distinguish it from a clean unchanged classification.
                props["borderline_reclassified"] = True
                border_recls["to_unchanged"] += 1
    summary.qa["borderline_modified_reclassifications"] = dict(border_recls)
    summary.qa["borderline_modified_margin"] = round(border_margin, 4)
    # NOTE: `summary.qa["change_counts"] = dict(counts)` runs unconditionally
    # below, so the reclassifications above are picked up regardless of
    # whether border_recls["to_unchanged"] is positive.

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
