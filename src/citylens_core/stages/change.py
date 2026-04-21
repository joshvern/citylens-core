from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


def _min_component_area_px() -> int:
    """Connected components below this pixel area are dropped as noise.

    Tunable via CITYLENS_CHANGE_MIN_AREA_PX. Default 8 — kills single-pixel
    artifacts from morphology/thresholding without swallowing small buildings
    at 1024x1024 resolution (a 5x5 building is still 25 px).
    """
    raw = os.getenv("CITYLENS_CHANGE_MIN_AREA_PX", "").strip()
    try:
        value = int(raw) if raw else 8
    except ValueError:
        return 8
    return max(0, value)


def _affine_identity():
    from rasterio.transform import Affine

    return Affine.identity()


def _polygonize_mask(
    mask: Any,
    *,
    transform: Any | None,
    min_area_px: int | None = None,
) -> list[list[list[list[float]]]]:
    """Trace a boolean mask into one ring-list per connected component.

    Returns a list of Polygon `coordinates` values (each is a list of rings).
    The outer ring comes first, followed by interior holes (if any).
    Coordinates are in (x, y) — either pixel space (if transform is None)
    or projected world units (if transform is provided).
    """
    import numpy as np
    from rasterio.features import shapes

    m = np.asarray(mask).astype(bool)
    if m.size == 0 or not m.any():
        return []

    if min_area_px is None:
        min_area_px = _min_component_area_px()

    # `shapes` needs a transform; if none given, use identity so we stay in
    # pixel coordinates.
    tr = transform if transform is not None else _affine_identity()

    polygons: list[list[list[list[float]]]] = []
    m_uint8 = m.astype("uint8")
    for geom, value in shapes(m_uint8, mask=m, transform=tr):
        if int(value) != 1:
            continue
        if geom.get("type") != "Polygon":
            continue
        coordinates = geom.get("coordinates") or []
        if not coordinates:
            continue

        # Filter tiny specks by computing the polygon's pixel area.
        # In pixel space this is trivial; in world space we compute the ring
        # area analytically (shoelace) and bail if it's below the threshold
        # scaled by pixel size.
        outer = coordinates[0]
        try:
            area_world = abs(
                sum(
                    (outer[i][0] + outer[i + 1][0])
                    * (outer[i + 1][1] - outer[i][1])
                    for i in range(len(outer) - 1)
                )
                / 2.0
            )
        except Exception:
            area_world = 0.0

        if transform is None:
            if area_world < float(min_area_px):
                continue
        else:
            # Transform.a is x-pixel size, transform.e is y-pixel size (usually negative).
            try:
                px_area = abs(float(transform.a) * float(transform.e))
            except Exception:
                px_area = 1.0
            min_world_area = max(px_area, 1e-9) * float(min_area_px)
            if area_world < min_world_area:
                continue

        polygons.append([list(ring) for ring in coordinates])

    return polygons


def _build_feature(
    *,
    kind: str,
    coordinates: list[list[list[float]]],
    imagery_year: int,
    baseline_year: int,
    crs_value: str,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "kind": kind,
            "imagery_year": imagery_year,
            "baseline_year": baseline_year,
            "crs": crs_value,
        },
        "geometry": {"type": "Polygon", "coordinates": coordinates},
    }


def stage_change(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
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
        transform = None
        crs_value = "pixel"
    else:
        crs_value = str(crs)

    import numpy as np

    im = np.asarray(imagery_mask).astype(bool)
    base = np.asarray(baseline_mask).astype(bool)
    added = np.logical_and(im, np.logical_not(base))
    removed = np.logical_and(base, np.logical_not(im))
    change_mask = np.logical_or(added, removed)

    features: list[dict[str, Any]] = []
    for kind, m in ("added", added), ("removed", removed):
        polygons = _polygonize_mask(m, transform=transform)
        for coordinates in polygons:
            features.append(
                _build_feature(
                    kind=kind,
                    coordinates=coordinates,
                    imagery_year=request.imagery_year,
                    baseline_year=request.baseline_year,
                    crs_value=crs_value,
                )
            )

    feature_collection = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(feature_collection, indent=2))
    return {**ctx, "change_path": out_path, "change_mask": change_mask.astype(np.uint8)}
