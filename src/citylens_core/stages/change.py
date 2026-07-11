from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..io.geo import binary_mask_stats
from ..models import CitylensRequest, PipelineSummary
from ._polygon_smoothing import (
    estimate_pixel_tolerance_in_world_units,
    simplify_polygon_coords,
)
from ._registration import RegistrationResult, apply_shift, estimate_alignment
from ._surface_change import (
    _load_rgb,
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
    """Drop 'added' components smaller than this area in square meters.

    Default 60 m². History: the default was 100 m² while `_pixel_area_m2`
    overstated EPSG:3857 areas by ~sec²(lat) (≈1.74× at NYC latitude), so the
    gate actually fired at ~57 true-ground m². With areas now latitude-
    corrected, 60 m² preserves that effective behavior AND keeps small NYC
    infill rowhouses (~90-140 m² footprints) safely above the gate.
    """
    return _float_env("CITYLENS_CHANGE_MIN_AREA_M2", 60.0)


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


def _added_reject_border_touching() -> bool:
    """Reject incomplete/noisy discovery components clipped by the tile.

    Automatic segmentation often emits one large road, water, or background
    region connected to an image edge.  With baseline-epoch LiDAR that flat
    region can otherwise look like strong evidence of new construction.
    Border-clipped buildings are not reliable complete footprints either, so
    the conservative production default is to reject them.  The switch keeps
    controlled ablations possible.
    """
    return _bool_env("CITYLENS_CHANGE_ADDED_REJECT_BORDER_TOUCHING", True)


def _current_source_added_confidence() -> float:
    """Confidence for a dated semantic footprint in baseline-empty space."""
    return _float_env("CITYLENS_CHANGE_CURRENT_SOURCE_ADDED_CONFIDENCE", 0.9)


def _current_source_modified_confidence() -> float:
    """Confidence for a dated footprint overlapping/replacing a baseline."""
    return _float_env("CITYLENS_CHANGE_CURRENT_SOURCE_MODIFIED_CONFIDENCE", 0.85)


def _added_max_baseline_epoch_height_m() -> float:
    """LiDAR height gate on 'added' components — BASELINE-EPOCH semantics.

    The production LiDAR is baseline-epoch (2017 NYS TopoBathymetric) while
    the imagery is current (2024). For a candidate new building that means:

      - LiDAR ~ground level inside the component  →  the parcel was EMPTY at
        the baseline epoch. Combined with a building-shaped 2024 SAM2 mask
        (and the ExG vegetation reject), that is POSITIVE evidence of new
        construction — accept as 'added' with boosted confidence.
      - LiDAR tall inside the component  →  something already stood there at
        the baseline epoch (tree canopy in first-return LiDAR, or a building
        missing from the footprints GDB). NOT new construction — emit as
        'candidate_added' for review rather than 'added'.

    This is the inversion of the original gate, which assumed current-epoch
    LiDAR and rejected flat components as "too short" — with 2017 LiDAR that
    guaranteed rejecting exactly the new buildings the gate exists to find.

    Value: a component whose baseline-epoch height-above-ground is BELOW this
    threshold counts as "was flat". Default 2 m. The old env name
    `CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M` is honored as a fallback for
    existing deploys but its meaning is now "max baseline-epoch height".
    """
    raw = os.getenv("CITYLENS_CHANGE_ADDED_MAX_BASELINE_HEIGHT_M", "").strip()
    if raw:
        return _float_env("CITYLENS_CHANGE_ADDED_MAX_BASELINE_HEIGHT_M", 2.0)
    return _float_env("CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M", 2.0)


def _added_exg_vegetation_threshold() -> float:
    """Excess-green index (ExG = 2G − R − B on 0-255 RGB) above which an
    'added' candidate is rejected as vegetation.

    The baseline-epoch LiDAR gate can no longer reject trees by height in the
    current epoch (2017 LiDAR says nothing about a tree that grew by 2024),
    so vegetation discrimination moves to the imagery itself: healthy canopy
    has strongly positive ExG, roofs (gray/black/brown/white) sit near or
    below zero. Median ExG over the component keeps single green pixels from
    dominating. Default 30.0 — comfortably above roofing materials, below
    healthy foliage. Set <= 0 to disable. Only applies when the orthophoto
    RGB is loadable at mask shape."""
    return _float_env("CITYLENS_CHANGE_ADDED_EXG_VEG_THRESHOLD", 30.0)


def _added_strong_confidence() -> float:
    """Confidence for 'added' components confirmed by the baseline-epoch
    LiDAR gate (parcel was flat at baseline epoch + building-shaped now +
    not vegetation). Two independent signals agree, so this sits above the
    generic added confidence (0.7)."""
    return _float_env("CITYLENS_CHANGE_ADDED_STRONG_CONFIDENCE", 0.85)


def _added_height_percentile() -> float:
    """Percentile of LiDAR z-values within a candidate 'added' footprint to
    compare against the ground plane. Using the 75th percentile (instead
    of the max) makes the gate robust against single tall trees/antennas
    poking through an otherwise-flat patch of grass."""
    return _float_env("CITYLENS_CHANGE_ADDED_HEIGHT_PERCENTILE", 75.0)


def _added_baseline_dilate_px() -> int:
    """Pixel-radius dilation of the baseline mask used by the courtyard +
    near-baseline filters on 'added' candidates.

    The 1-pixel `_one_step_dilate` overlap filter only catches candidates
    that physically TOUCH a baseline footprint — but two failure modes
    sit further out:

    1. Courtyards / lightwells: 2-10 px gap between rasterized GDB
       footprints (sidewalk moat).
    2. Existing buildings the matcher missed because per-feature
       registration didn't apply: imagery mask is offset by 2-7m from the
       baseline, so the imagery component falls in apparently-empty space
       that's actually right next to a baseline footprint.

    Default 24 px ≈ 7 m at 0.3 m/px (12 m at 0.5 m/px). Wide enough to
    span the typical NYC alignment-error budget AND most rasterized
    courtyard gaps; narrow enough that a genuine new building one parcel
    over (≥10m clearance) is still flagged. Bump to 32+ in dense
    neighborhoods if false-positive 'added' polygons are still slipping
    through; drop back to 8 in sparse / rural data where baselines are
    sparse and any non-overlap is a real new structure.
    """
    return _int_env("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", 24)


def _added_max_inside_dilation_frac() -> float:
    """Reject 'added' candidates that are mostly *inside* the wide-dilated
    baseline.

    The centroid-only filter checks one pixel and misses large blobs whose
    centroid happens to fall in a gap between baselines (e.g., a 30×50m
    imagery component sitting on top of three baseline footprints — its
    centroid lands in the small interior void between them). Rejecting
    when ≥50% of the candidate's pixels lie inside the dilated baseline
    catches those cases without flipping legitimate new construction
    (which has ≪50% overlap with any nearby baseline dilation).

    Set to 1.0 to disable.
    """
    return _float_env("CITYLENS_CHANGE_ADDED_MAX_INSIDE_DILATION_FRAC", 0.5)


def _demolished_max_height_m() -> float:
    """LiDAR-validated demolished-rescue threshold (see
    `_demolished_rescue_lidar_epoch` for when the rescue applies at all).

    When the rescue is active and LiDAR shows a structure standing inside a
    baseline footprint (75th-percentile height above ground exceeds this),
    a would-be "demolished" call is downgraded to "modified" — SAM2 missed
    the building (dark roof, shadow) but LiDAR confirms it's there.

    Default 3.0 m: taller than a parked truck or hedge, shorter than every
    real one-story building.
    """
    return _float_env("CITYLENS_CHANGE_DEMOLISHED_MAX_HEIGHT_M", 3.0)


def _demolished_rescue_lidar_epoch() -> str:
    """Which acquisition epoch the LiDAR grid belongs to, for the
    demolished→modified rescue. One of:

      - "baseline" (default): LiDAR is contemporaneous with the BASELINE
        (production reality: 2017 NYS TopoBathymetric vs 2024 imagery). A
        standing structure in baseline-epoch LiDAR is EXPECTED for a real
        2017→2024 demolition — the old building was there in 2017 whether or
        not it was demolished later — so the rescue has zero discriminative
        power and is DISABLED. (Previously it ran anyway and downgraded
        essentially every genuine demolition to "modified".)
      - "current": LiDAR is contemporaneous with the CURRENT imagery. A
        standing structure genuinely contradicts "demolished", so the
        original rescue applies.

    Known trade-off under "baseline": SAM2 dark-roof misses can surface as
    false "demolished" calls again; those carry their IoU-derived confidence
    and remain auditable.
    """
    raw = os.getenv("CITYLENS_CHANGE_DEMOLISHED_RESCUE_LIDAR_EPOCH", "").strip().lower()
    return raw if raw in ("baseline", "current") else "baseline"


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


def _mask_touches_border(mask: Any) -> bool:
    """Whether any True pixel lies on the image boundary."""
    import numpy as np

    m = np.asarray(mask).astype(bool)
    if m.ndim != 2 or not m.any():
        return False
    return bool(
        m[0, :].any()
        or m[-1, :].any()
        or m[:, 0].any()
        or m[:, -1].any()
    )


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
    confidently convert (e.g. pixel-space only or unknown CRS).

    Web-Mercator (EPSG:3857) "meters" are not ground meters: the map scale
    is sec(lat), so a naive a*e product overstates ground area by sec²(lat)
    — ~1.74× at NYC's 40.7°N. We derive the latitude from the transform's
    y-origin and correct by cos²(lat). Every consumer of `area_m2` (the
    min-area gate, the published change features, the parcel-intel moat)
    depends on this being true ground area.
    """
    import math

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
    if s in ("pixel", ""):
        return 0.0
    cleaned = s.replace("-", "").replace("_", "")
    if ("ftus" in cleaned or "2263" in cleaned or cleaned.endswith("ft")) and "meter" not in s:
        # NY State Plane (EPSG:2263) & friends: US survey feet.
        # 1 ft² ≈ 0.09290341 m². State Plane scale error is <0.01% — no
        # latitude correction needed.
        return px * 0.0929034116
    if "3857" in cleaned or "pseudomercator" in cleaned or "webmercator" in cleaned:
        try:
            # Invert the spherical-Mercator y of the raster's top edge to
            # latitude: lat = atan(sinh(y / R)). Top-vs-center differs by
            # <0.005° over a 512 m AOI — negligible for cos².
            lat = math.atan(math.sinh(float(transform.f) / 6378137.0))
            return px * math.cos(lat) ** 2
        except Exception:
            return px
    return px


def _feature(
    *,
    change_type: str,
    coordinates: Any,
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
    geometry_type: str = "Polygon",
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
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


# ----------------------------------------------------------------------
# Per-source-feature classification helpers
# ----------------------------------------------------------------------


def _load_source_features(path: Path) -> list[dict[str, Any]] | None:
    """Load a local GeoJSON FeatureCollection without any external I/O.

    ``None`` means absent/invalid and enables the caller's fallback. A valid
    empty collection returns ``[]`` because an authoritative source may
    intentionally attest that no buildings exist in the AOI.
    """
    gj_path = Path(path)
    if not gj_path.exists():
        return None
    try:
        payload = json.loads(gj_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "FeatureCollection":
        return None
    feats = payload.get("features")
    if not isinstance(feats, list):
        return None
    return [f for f in feats if isinstance(f, dict) and f.get("geometry")]


def _load_baseline_source_features(work_dir: Path) -> list[dict[str, Any]] | None:
    """Load staged baseline building features when available."""
    return _load_source_features(work_dir / "baseline_footprints.geojson")


def _load_current_source_features(work_dir: Path) -> list[dict[str, Any]] | None:
    """Load optional semantic current buildings from the local work dir."""
    return _load_source_features(work_dir / "current_footprints.geojson")


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


def _construction_year(source_props: Any) -> int | None:
    """Normalize a source construction year, rejecting bools/junk."""
    if not isinstance(source_props, dict):
        return None
    raw = source_props.get("construction_year")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        year = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 9999 else None


def _extract_current_source_provenance(source_props: Any) -> dict[str, Any]:
    """Forward the stable semantic-current-footprint provenance contract."""
    if not isinstance(source_props, dict):
        return {"current_footprint_semantic": True}
    keep = (
        "last_status_type",
        "geom_source",
        "base_bbl",
        "mappluto_bbl",
        "source_dataset",
    )
    out = {k: source_props[k] for k in keep if k in source_props}
    year = _construction_year(source_props)
    if year is not None:
        out["construction_year"] = year
    out["current_footprint_semantic"] = True
    return out


def _best_local_iou(
    im: Any,
    fp_roi: Any,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    max_shift: int = 2,
    min_gain: float = 0.02,
) -> tuple[float, tuple[int, int] | None]:
    """Per-footprint local registration refinement.

    The single global phase-correlation shift is dominated by the largest
    footprints in the tile; small buildings can carry a residual local
    misalignment that reads as a spurious IoU drop (false "modified"). Slide
    the IMAGERY window ±`max_shift` px around the footprint's ROI and take
    the best IoU — adopted only when it beats the unshifted IoU by
    `min_gain`, so noise can't inflate scores.

    Implementation note: slides the imagery *window* (O(ROI px) per shift,
    ~5e7 boolean ops for 200 footprints at ±2) — never roll the full-tile
    mask, which is ~100× more work.

    Returns (iou, (dy, dx) or None when the unshifted IoU stands).
    """
    import numpy as np

    h, w = im.shape
    fp = np.asarray(fp_roi)
    fp_any = bool(fp.any())

    def _iou_at(u: int, v: int) -> float:
        sy0, sy1 = y0 + u, y1 + u
        sx0, sx1 = x0 + v, x1 + v
        # Clamp the shifted window to the image; keep the footprint ROI
        # aligned by trimming the same rows/cols.
        ty0, ty1 = max(sy0, 0), min(sy1, h)
        tx0, tx1 = max(sx0, 0), min(sx1, w)
        if ty0 >= ty1 or tx0 >= tx1:
            return 0.0
        im_win = im[ty0:ty1, tx0:tx1]
        fp_win = fp[ty0 - sy0 : (ty1 - sy0), tx0 - sx0 : (tx1 - sx0)]
        inter = int(np.logical_and(fp_win, im_win).sum())
        union = int(np.logical_or(fp_win, im_win).sum())
        return float(inter) / float(union) if union > 0 else 0.0

    base_iou = _iou_at(0, 0)
    if not fp_any or max_shift <= 0:
        return base_iou, None

    best_iou = base_iou
    best_shift: tuple[int, int] | None = None
    for u in range(-max_shift, max_shift + 1):
        for v in range(-max_shift, max_shift + 1):
            if u == 0 and v == 0:
                continue
            iou = _iou_at(u, v)
            if iou > best_iou:
                best_iou = iou
                best_shift = (u, v)

    if best_shift is not None and best_iou >= base_iou + min_gain:
        return best_iou, best_shift
    return base_iou, None


def _local_registration_max_shift_px() -> int:
    """Per-footprint local refinement window (± this many px) applied on top
    of the global registration shift. 0 disables. Default 2 — the residual
    after the ±4 px global correction is sub-3 px in observed NYS data."""
    return _int_env("CITYLENS_CHANGE_LOCAL_REGISTRATION_MAX_SHIFT_PX", 2)


def _local_registration_min_iou_gain() -> float:
    """Minimum IoU improvement over the unshifted score before a local
    per-footprint shift is adopted. Guards against max-over-shifts noise
    inflation masking small real modifications."""
    return _float_env("CITYLENS_CHANGE_LOCAL_REGISTRATION_MIN_IOU_GAIN", 0.02)


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

    # A staged current-footprint FeatureCollection is the preferred semantic
    # source for "does a building exist now?" It is already in the ortho CRS.
    # Keep `im` as the SAM mask for pipeline QA/preview consumers, but classify
    # baseline presence against this vector-derived mask when available.
    current_source_features = _load_current_source_features(work_dir)
    current_footprints_mask = ctx.get("current_footprints_mask")
    if (
        current_footprints_mask is None
        and current_source_features is not None
        and transform is not None
    ):
        current_union = np.zeros_like(im, dtype=bool)
        for current_feature in current_source_features:
            current_union |= _rasterize_single_geom(
                current_feature.get("geometry"),
                out_shape=im.shape,
                transform=transform,
            )
        current_footprints_mask = current_union

    if current_footprints_mask is not None:
        current_presence = np.asarray(current_footprints_mask).astype(bool)
        if current_presence.shape != im.shape:
            raise RuntimeError(
                "current footprints mask shape "
                f"{current_presence.shape} does not match imagery mask {im.shape}"
            )
        classification_im = current_presence
        summary.qa["current_footprints_used"] = True
        summary.qa["current_presence_source"] = "current_footprints"
        summary.qa["current_footprints_mask"] = binary_mask_stats(current_presence)
    else:
        classification_im = im
        summary.qa["current_footprints_used"] = False
        summary.qa["current_presence_source"] = "sam2"
    summary.qa["current_footprint_feature_count"] = (
        len(current_source_features) if current_source_features is not None else 0
    )

    # This discovery mask is separate from baseline-presence classification:
    # semantic current footprints (preferred) or prompted SAM drive existing
    # footprints, while AMG is only a generic-added fallback. Auto mode aliases
    # its primary mask here without paying for duplicate automatic inference.
    discovery_mask = ctx.get("refined_added_discovery_mask")
    added_mask_source = "refined_added_discovery_mask"
    if discovery_mask is None:
        discovery_mask = ctx.get("added_discovery_mask")
        added_mask_source = "added_discovery_mask"
    if discovery_mask is None:
        discovery_mask = imagery_mask
        added_mask_source = "imagery_mask"
    added_im = np.asarray(discovery_mask).astype(bool)
    if added_im.shape != im.shape:
        raise RuntimeError(
            f"discovery mask shape {added_im.shape} does not match imagery mask {im.shape}"
        )
    summary.qa["added_mask_source"] = added_mask_source

    # ------------------------------------------------------------------
    # Sub-pixel registration (#1). Estimate one global (dy, dx) shift
    # that aligns the baseline mask to the current mask. Recovers the
    # per-footprint IoU lost to acquisition-year registration error
    # (typically 0.5–2 px between NYS Orthos baseline and current).
    # If the phase-correlation peak isn't confident enough or the IoU
    # gain is trivial, the shift is NOT applied — defensive default.
    # ------------------------------------------------------------------
    if current_footprints_mask is not None:
        # Both semantic layers are already on the orthophoto grid. Preserve
        # source geometry exactly and avoid an unnecessary full-tile FFT.
        semantic_intersection = int(np.logical_and(base, classification_im).sum())
        semantic_union = int(np.logical_or(base, classification_im).sum())
        semantic_iou = (
            float(semantic_intersection) / float(semantic_union)
            if semantic_union
            else 1.0
        )
        registration = RegistrationResult(
            dy=0.0,
            dx=0.0,
            confidence=1.0,
            iou_before=semantic_iou,
            iou_after=semantic_iou,
            accepted=False,
        )
    else:
        registration = estimate_alignment(
            base,
            classification_im,
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
        "mode": (
            "semantic_exact"
            if current_footprints_mask is not None
            else "image_registration"
        ),
        # True when the estimated shift hit the ± cap — the real
        # misregistration may be larger than what was corrected; expect
        # residual false 'modified' on such tiles.
        "saturated": bool(
            max(abs(registration.dy), abs(registration.dx))
            >= _registration_max_shift_px()
        ),
    }
    if registration.accepted:
        base = apply_shift(base, dy=registration.dy, dx=registration.dx)
        # Keep the rasterized-footprints map in sync if the per-source
        # path later uses it and for downstream ctx readers.
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

    # Dated semantic current footprints are direct change evidence. Build the
    # event records before baseline classification so a replacement can
    # suppress the stale baseline output it supersedes (avoids emitting both
    # "unchanged old footprint" and "modified replacement" on one building).
    semantic_current_events: list[dict[str, Any]] = []
    semantic_current_rejected = {"too_small": 0, "border_touching": 0}
    semantic_modified_mask = np.zeros_like(base, dtype=bool)
    semantic_current_usable = bool(
        current_source_features is not None
        and current_footprints_mask is not None
        and transform is not None
    )
    generic_added_fallback_used = not semantic_current_usable
    if semantic_current_usable:
        added_mask_source = "current_footprints"
        summary.qa["added_mask_source"] = added_mask_source
    summary.qa["generic_added_fallback_used"] = generic_added_fallback_used
    if semantic_current_usable:
        for current_feature in current_source_features:
            current_props = current_feature.get("properties")
            construction_year = _construction_year(current_props)
            if (
                construction_year is None
                or construction_year <= request.baseline_year
                or construction_year > request.imagery_year
            ):
                continue
            current_geom = current_feature.get("geometry")
            if not isinstance(current_geom, dict):
                continue
            geometry_type = str(current_geom.get("type", "")).strip()
            coordinates = current_geom.get("coordinates")
            if geometry_type not in ("Polygon", "MultiPolygon") or not coordinates:
                continue
            current_mask = _rasterize_single_geom(
                current_geom,
                out_shape=base.shape,
                transform=transform,
            )
            current_area_px = int(current_mask.sum())
            if current_area_px < 1:
                continue
            if _mask_touches_border(current_mask):
                # Even with padded source queries, an edge-clipped geometry
                # is incomplete evidence for a change claim.
                semantic_current_rejected["border_touching"] += 1
                continue
            if current_area_px < min_area_px:
                # Apply the same commercial noise floor as generic added
                # components before this event can suppress a baseline row.
                semantic_current_rejected["too_small"] += 1
                continue
            baseline_overlap_px = int(np.logical_and(current_mask, base).sum())
            baseline_overlap_fraction = (
                float(baseline_overlap_px) / float(current_area_px)
            )
            semantic_change_type = (
                "added"
                if baseline_overlap_fraction <= added_overlap_cap
                else "modified"
            )
            if semantic_change_type == "modified":
                semantic_modified_mask |= current_mask
            semantic_current_events.append(
                {
                    "change_type": semantic_change_type,
                    "area_px": current_area_px,
                    "baseline_overlap_fraction": baseline_overlap_fraction,
                    "geometry_type": geometry_type,
                    "coordinates": coordinates,
                    "properties": current_props,
                }
            )

    semantic_baseline_outputs_suppressed = 0
    baseline_edge_changes_skipped = {"modified": 0, "demolished": 0}

    def _superseded_by_semantic_current(footprint_mask: Any) -> bool:
        """Whether a dated replacement materially covers this baseline."""
        if not semantic_modified_mask.any():
            return False
        fp = np.asarray(footprint_mask).astype(bool)
        fp_area = int(fp.sum())
        if fp_area < 1:
            return False
        overlap = int(np.logical_and(fp, semantic_modified_mask).sum())
        return overlap > 0 and (
            float(overlap) / float(fp_area)
        ) >= added_overlap_cap

    # LiDAR-validated demolished rescue. Pulled up here so both the
    # per-source-feature path and the legacy component-labeled fallback can
    # use it. `None` ⇒ no LiDAR available, rescue disabled (legacy
    # behavior).
    lidar_heights = ctx.get("lidar_heights")
    lidar_ground_z = ctx.get("lidar_ground_z")
    demolished_max_h = _demolished_max_height_m()
    demolished_pctl = _demolished_height_percentile()
    # Epoch gate: with baseline-epoch LiDAR (production: 2017 grid vs 2024
    # imagery) a standing structure is EXPECTED for a genuine demolition, so
    # the rescue carries no signal and previously converted essentially every
    # real demolition into "modified". Only rescue with current-epoch LiDAR.
    demolished_rescue_enabled = _demolished_rescue_lidar_epoch() == "current"
    summary.qa.setdefault("demolished_downgraded_to_modified", 0)
    summary.qa["demolished_rescue_lidar_epoch"] = _demolished_rescue_lidar_epoch()

    def _classify_with_lidar_rescue(iou: float, footprint_mask) -> str:
        """Apply IoU bands, then (current-epoch LiDAR only) downgrade
        demolished→modified if LiDAR shows a structure still standing
        inside the baseline footprint."""
        if iou >= unchanged_thresh:
            return "unchanged"
        if iou >= modified_thresh:
            return "modified"
        if (
            demolished_rescue_enabled
            and lidar_heights is not None
            and lidar_ground_z is not None
        ):
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

    local_reg_max_shift = (
        0
        if current_footprints_mask is not None
        else _local_registration_max_shift_px()
    )
    local_reg_min_gain = _local_registration_min_iou_gain()
    local_reg_applied = 0

    if source_features is not None and not pixel_space_only:
        summary.qa["change_source"] = "per_source_feature"
        h, w = classification_im.shape
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

            scoring_mask = (
                apply_shift(single_mask, dy=registration.dy, dx=registration.dx).astype(bool)
                if registration.accepted
                else single_mask
            )
            if not scoring_mask.any():
                scoring_mask = single_mask
            if _superseded_by_semantic_current(scoring_mask):
                semantic_baseline_outputs_suppressed += 1
                continue

            # IoU measured within the footprint's bbox (pad 10%) so a
            # single large SAM2 blob can't swallow every neighbor. On top of
            # the global shift, a small per-footprint refinement recovers
            # residual local misalignment (the global shift is dominated by
            # the largest buildings and can leave small footprints offset).
            ys, xs = np.where(scoring_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            pad_y = max(1, (y1 - y0) // 10)
            pad_x = max(1, (x1 - x0) // 10)
            y0 = max(0, y0 - pad_y)
            x0 = max(0, x0 - pad_x)
            y1 = min(h, y1 + pad_y)
            x1 = min(w, x1 + pad_x)
            fp_roi = scoring_mask[y0:y1, x0:x1]
            iou, local_shift = _best_local_iou(
                classification_im,
                fp_roi,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                max_shift=local_reg_max_shift,
                min_gain=local_reg_min_gain,
            )
            if local_shift is not None:
                local_reg_applied += 1

            change_type = _classify_with_lidar_rescue(iou, scoring_mask)
            if change_type in baseline_edge_changes_skipped and _mask_touches_border(
                scoring_mask
            ):
                baseline_edge_changes_skipped[change_type] += 1
                continue
            counts[change_type] += 1

            area_m2_val = float(single_area_px) * px_area_m2 if px_area_m2 > 0 else None
            provenance = _extract_source_provenance(src_feat.get("properties"))
            if local_shift is not None:
                # Audit trail: this footprint's IoU came from a locally
                # refined alignment, not the raw global registration.
                provenance = {**provenance, "local_shift_px": list(local_shift)}
            # Sample 95th-pct LiDAR height inside the footprint so each output
            # feature carries its own building height. Reused by UIs and by
            # the LOD1 mesh stage. None when LiDAR isn't available.
            feature_height_m: float | None = None
            if lidar_heights is not None and lidar_ground_z is not None:
                feature_height_m = _lidar_height_at_mask(
                    mask=scoring_mask,
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
                    footprint_mask=scoring_mask,
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
            if _superseded_by_semantic_current(comp):
                semantic_baseline_outputs_suppressed += 1
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
            iou, local_shift = _best_local_iou(
                classification_im,
                comp_roi,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                max_shift=local_reg_max_shift,
                min_gain=local_reg_min_gain,
            )
            if local_shift is not None:
                local_reg_applied += 1

            change_type = _classify_with_lidar_rescue(iou, comp)
            if change_type in baseline_edge_changes_skipped and _mask_touches_border(
                comp
            ):
                baseline_edge_changes_skipped[change_type] += 1
                continue
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
    # 2. Emit dated semantic current-footprint events.
    # ------------------------------------------------------------------
    semantic_current_counts = {"added": 0, "modified": 0}
    for semantic_event in semantic_current_events:
        semantic_change_type = str(semantic_event["change_type"])
        semantic_current_counts[semantic_change_type] += 1
        counts[semantic_change_type] += 1
        semantic_area_m2 = (
            float(semantic_event["area_px"]) * px_area_m2
            if px_area_m2 > 0
            else None
        )
        semantic_props = _extract_current_source_provenance(
            semantic_event.get("properties")
        )
        semantic_props.update(
            {
                "baseline_overlap_fraction": round(
                    float(semantic_event["baseline_overlap_fraction"]), 4
                ),
                "semantic_change_basis": "construction_year",
            }
        )
        semantic_confidence = (
            _current_source_added_confidence()
            if semantic_change_type == "added"
            else _current_source_modified_confidence()
        )
        # Preserve one source feature as one event. In particular, a
        # MultiPolygon keeps one total area and increments counts once rather
        # than emitting N Polygon rows that each claim the whole-feature area.
        features.append(
            _feature(
                change_type=semantic_change_type,
                coordinates=semantic_event["coordinates"],
                area_m2=semantic_area_m2,
                baseline_iou=None,
                imagery_year=request.imagery_year,
                baseline_year=request.baseline_year,
                crs_value=crs_value,
                extra_props=semantic_props,
                confidence=semantic_confidence,
                geometry_type=str(semantic_event["geometry_type"]),
            )
        )
    summary.qa["semantic_current_change_counts"] = semantic_current_counts
    summary.qa["semantic_current_rejected"] = semantic_current_rejected
    summary.qa["semantic_baseline_outputs_suppressed"] = int(
        semantic_baseline_outputs_suppressed
    )

    # ------------------------------------------------------------------
    # 3. Generic added fallback — automatic/prompted components that don't
    #    overlap a baseline. A usable semantic current collection is complete
    #    for the AOI and disables this noisier lane, preventing duplicates.
    # ------------------------------------------------------------------
    added_max_baseline_h = _added_max_baseline_epoch_height_m()
    added_height_pctl = _added_height_percentile()
    added_baseline_dilate_px = _added_baseline_dilate_px()
    added_max_inside_dilation_frac = _added_max_inside_dilation_frac()
    added_exg_threshold = _added_exg_vegetation_threshold()
    reject_border_touching = (
        _added_reject_border_touching() and added_mask_source != "imagery_mask"
    )
    added_reject_reasons = {
        "too_small": 0,
        "border_touching": 0,
        "baseline_overlap": 0,
        "centroid_near_baseline": 0,
        "majority_inside_baseline_dilation": 0,
        "vegetation": 0,
        "preexisting_structure_emitted_as_candidate": 0,
        "no_lidar_coverage_emitted_as_candidate": 0,
    }

    # Orthophoto RGB for the ExG vegetation reject. Loaded directly (NOT via
    # `surface_images`, which is None whenever the baseline is a one-band
    # footprint mask — the production configuration). Soft-skips when the
    # ortho path is absent or the shape mismatches (unit-test fixtures).
    ortho_rgb = None
    if generic_added_fallback_used and added_exg_threshold > 0:
        _ortho_path = ctx.get("orthophoto_path")
        if _ortho_path is not None:
            _rgb = _load_rgb(Path(_ortho_path))
            if _rgb is not None and _rgb.shape[:2] == im.shape:
                ortho_rgb = _rgb

    # Pre-compute the wide baseline dilation used by the courtyard /
    # centroid-near-baseline filter. Built once outside the per-component
    # loop because each step is O(H*W) and we'd otherwise pay it `added_n`
    # times.
    baseline_dilated_wide = (
        _n_step_dilate(base, added_baseline_dilate_px)
        if generic_added_fallback_used and added_baseline_dilate_px > 0
        else None
    )

    added_pixels = (
        np.logical_and(added_im, np.logical_not(base))
        if generic_added_fallback_used
        else np.zeros_like(base, dtype=bool)
    )
    added_labels, added_n = _label_components(added_pixels)
    for comp_id in range(1, added_n + 1):
        comp = added_labels == comp_id
        comp_area_px = int(comp.sum())
        if comp_area_px < min_area_px:
            added_reject_reasons["too_small"] += 1
            continue

        # AMG commonly yields a huge road/background region connected to the
        # tile edge.  It is both an obvious false building and an incomplete
        # polygon, so reject it before the more expensive per-component gates.
        if reject_border_touching and _mask_touches_border(comp):
            added_reject_reasons["border_touching"] += 1
            continue

        # Reject slivers along existing baseline buildings.
        overlap_touching = int(np.logical_and(comp, _one_step_dilate(base)).sum())
        overlap_fraction = float(overlap_touching) / float(comp_area_px) if comp_area_px else 1.0
        if overlap_fraction > added_overlap_cap:
            added_reject_reasons["baseline_overlap"] += 1
            continue

        # Reject courtyards / lightwells / alignment-error twins: any
        # candidate whose centroid lies inside the wide-dilated baseline
        # mask is almost certainly either an interior gap between
        # buildings (rasterized GDB footprints snap to whole-pixel edges
        # and there's a small road/sidewalk gap) or an existing building
        # the matcher missed because per-feature registration didn't
        # apply (imagery mask is offset by a few meters from the baseline
        # footprint, so the unmatched imagery component sits a few pixels
        # away from "its" baseline polygon).
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

            # Centroid-only is one pixel — a 30×50m imagery blob sitting
            # on top of three baseline footprints can have its centroid
            # land in the small void between them and pass the check
            # above. Reject when a majority of the component's pixels are
            # inside the wide-dilated baseline; legitimate new
            # construction has well under 50% overlap with any nearby
            # baseline dilation.
            if added_max_inside_dilation_frac < 1.0:
                inside_px = int(np.logical_and(comp, baseline_dilated_wide).sum())
                inside_frac = (
                    float(inside_px) / float(comp_area_px) if comp_area_px else 0.0
                )
                if inside_frac >= added_max_inside_dilation_frac:
                    added_reject_reasons["majority_inside_baseline_dilation"] += 1
                    continue

        # Vegetation reject on the imagery itself. The LiDAR gate below is
        # BASELINE-epoch (2017) and therefore says nothing about a tree that
        # grew or leafed out by the current epoch — ExG on the 2024 RGB is
        # the discriminator: healthy canopy is strongly green-excess, roofs
        # are not. Median over the component resists mixed pixels.
        if ortho_rgb is not None:
            comp_px = ortho_rgb[comp]
            if comp_px.size:
                px_f = comp_px.astype("float64")
                exg = float(
                    np.median(2.0 * px_f[:, 1] - px_f[:, 0] - px_f[:, 2])
                )
                if exg >= added_exg_threshold:
                    added_reject_reasons["vegetation"] += 1
                    continue

        # Baseline-EPOCH LiDAR gate (see _added_max_baseline_epoch_height_m):
        # the LiDAR grid is contemporaneous with the BASELINE (2017), so
        #   flat then + building-shaped now  →  built since baseline: strong
        #                                       'added' (two signals agree);
        #   tall then                       →  something already stood there
        #                                       (canopy / GDB-missing bldg):
        #                                       demote to 'candidate_added';
        #   no coverage                     →  'candidate_added' (unknown).
        added_height_above_ground_m: float | None = None
        emit_as_candidate = False
        candidate_reason: str | None = None
        strong_added = False
        if lidar_heights is not None and lidar_ground_z is not None:
            added_height_above_ground_m = _lidar_height_at_mask(
                mask=comp,
                lidar_heights=lidar_heights,
                lidar_ground_z=float(lidar_ground_z),
                percentile=added_height_pctl,
            )
            if added_height_above_ground_m is None:
                added_reject_reasons["no_lidar_coverage_emitted_as_candidate"] += 1
                emit_as_candidate = True
                candidate_reason = "no_lidar_coverage"
            elif added_height_above_ground_m < added_max_baseline_h:
                # Ground-level at the baseline epoch: positive evidence of
                # new construction.
                strong_added = True
            else:
                added_reject_reasons["preexisting_structure_emitted_as_candidate"] += 1
                emit_as_candidate = True
                candidate_reason = "preexisting_structure_in_baseline_lidar"

        if emit_as_candidate:
            change_type_emitted = "candidate_added"
            counts.setdefault("candidate_added", 0)
            counts["candidate_added"] += 1
            confidence_emitted = _candidate_added_confidence()
        else:
            change_type_emitted = "added"
            counts["added"] += 1
            confidence_emitted = (
                _added_strong_confidence()
                if strong_added
                else _classification_confidence(
                    change_type="added",
                    iou=None,
                    unchanged_thresh=unchanged_thresh,
                    modified_thresh=modified_thresh,
                )
            )

        area_m2_val = float(comp_area_px) * px_area_m2 if px_area_m2 > 0 else None
        # NOTE: for added/candidate_added features, height_m is the
        # BASELINE-epoch LiDAR height (near 0 for confirmed new builds).
        added_extra: dict[str, Any] = {}
        if candidate_reason is not None:
            added_extra["candidate_reason"] = candidate_reason
        if strong_added:
            added_extra["baseline_lidar_flat"] = True
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
                    extra_props=added_extra or None,
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
    # Median over NON-demolished baseline features only. Demolished
    # footprints legitimately score IoU≈0 — a genuinely redeveloping block
    # would otherwise drag the median down, lower the adaptive threshold,
    # and suppress real 'modified' calls on exactly the tiles that matter.
    # The COUNT gate stays on all baseline-derived samples so demolition-
    # heavy tiles keep the adaptive + borderline passes.
    median_samples = [
        f["properties"]["baseline_iou"]
        for f in features
        if isinstance(f.get("properties"), dict)
        and f["properties"].get("baseline_iou") is not None
        and f["properties"].get("change_type") != "demolished"
    ] or iou_samples
    unchanged_thresh_used = unchanged_thresh
    # Need a real sample size to trust the median — the adaptive logic is
    # only meaningful on a real-world tile with many baselines, never on
    # tests / synthetic fixtures where we'd otherwise classify the only
    # input feature against its own IoU.
    if len(iou_samples) >= _adaptive_threshold_min_samples():
        median_iou = float(np.median(median_samples))
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
    summary.qa["local_registration_applied"] = int(local_reg_applied)
    summary.qa["baseline_edge_changes_skipped"] = {
        **baseline_edge_changes_skipped,
        "total": int(sum(baseline_edge_changes_skipped.values())),
    }
    # Why the generic AMG/prompted added gate rejected candidates. Semantic
    # current-footprint additions bypass this noisy candidate lane and are
    # counted separately in semantic_current_change_counts.
    summary.qa["added_rejected"] = dict(added_reject_reasons)

    classification_change_mask = np.logical_or(
        np.logical_and(classification_im, np.logical_not(base)),
        np.logical_and(base, np.logical_not(classification_im)),
    )
    # Include current-only discovery pixels in the coarse downstream change
    # mask, while keeping them out of the baseline IoU classifier above.
    change_mask = np.logical_or(
        classification_change_mask,
        (
            np.logical_and(added_im, np.logical_not(base))
            if generic_added_fallback_used
            else np.zeros_like(base, dtype=bool)
        ),
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
    """Binary dilation by a `(2*steps+1) x (2*steps+1)` square kernel
    (identical result to `steps` iterated 3x3 dilations).

    Pure numpy (no scipy/skimage). A Chebyshev-ball dilation is separable
    into per-axis 1-D dilations, and each 1-D dilation is built by
    log-doubling shifted ORs: an accumulator with reach R, OR'd with copies
    shifted ±(R+1), has contiguous reach 2R+1. That's O(log steps) full-tile
    passes instead of O(steps) — the old 24-iteration loop over the whole
    tile was a measurable cost in the batch path.

    Used by the courtyard filter on 'added' candidates: dilating the
    baseline mask closes the typical road/sidewalk gap so candidates whose
    centroid lies inside the dilated mask can be rejected.
    """
    import numpy as np

    m = np.asarray(mask).astype(bool)
    if steps <= 0 or not m.any():
        return m

    def _dilate_axis(arr, axis: int, n: int):
        out = arr
        reach = 0
        while reach < n:
            s = min(reach + 1, n - reach)
            shifted_pos = np.zeros_like(out)
            shifted_neg = np.zeros_like(out)
            if axis == 0:
                shifted_pos[s:, :] = out[:-s, :]
                shifted_neg[:-s, :] = out[s:, :]
            else:
                shifted_pos[:, s:] = out[:, :-s]
                shifted_neg[:, :-s] = out[:, s:]
            out = out | shifted_pos | shifted_neg
            reach += s
        return out

    n = int(steps)
    return _dilate_axis(_dilate_axis(m, 0, n), 1, n)


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
