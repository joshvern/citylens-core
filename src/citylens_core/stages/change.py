from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


def _bbox_polygon(mask: Any) -> list[list[float]] | None:
    try:
        import numpy as np

        m = np.asarray(mask).astype(bool)
        if m.size == 0 or not m.any():
            return None
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        # GeoJSON ring in (x, y) pixel coordinates
        return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)], [float(x0), float(y0)]]
    except Exception:
        return None


def _ring_from_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    transform: Any | None = None,
) -> list[list[float]]:
    if transform is None:
        return [
            [float(x0), float(y0)],
            [float(x1), float(y0)],
            [float(x1), float(y1)],
            [float(x0), float(y1)],
            [float(x0), float(y0)],
        ]

    tl = transform * (float(x0), float(y0))
    tr = transform * (float(x1 + 1), float(y0))
    br = transform * (float(x1 + 1), float(y1 + 1))
    bl = transform * (float(x0), float(y1 + 1))
    return [
        [float(tl[0]), float(tl[1])],
        [float(tr[0]), float(tr[1])],
        [float(br[0]), float(br[1])],
        [float(bl[0]), float(bl[1])],
        [float(tl[0]), float(tl[1])],
    ]


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
        bbox = _bbox_polygon(m)
        if bbox is None:
            continue
        x0, y0 = int(bbox[0][0]), int(bbox[0][1])
        x1, y1 = int(bbox[2][0]), int(bbox[2][1])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": kind,
                    "imagery_year": request.imagery_year,
                    "baseline_year": request.baseline_year,
                    "crs": crs_value,
                },
                "geometry": {"type": "Polygon", "coordinates": [_ring_from_bbox(x0, y0, x1, y1, transform=transform)]},
            }
        )

    feature_collection = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(feature_collection, indent=2))
    return {**ctx, "change_path": out_path, "change_mask": change_mask.astype(np.uint8)}
