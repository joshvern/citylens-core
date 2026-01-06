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


def stage_change(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "change.geojson"

    imagery_mask = ctx.get("mask")
    baseline_mask = ctx.get("baseline_mask")
    if imagery_mask is None or baseline_mask is None:
        raise RuntimeError("change stage requires both imagery and baseline masks")

    import numpy as np

    im = np.asarray(imagery_mask).astype(bool)
    base = np.asarray(baseline_mask).astype(bool)
    added = np.logical_and(im, np.logical_not(base))
    removed = np.logical_and(base, np.logical_not(im))

    features: list[dict[str, Any]] = []
    for kind, m in ("added", added), ("removed", removed):
        ring = _bbox_polygon(m)
        if ring is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": kind,
                    "imagery_year": request.imagery_year,
                    "baseline_year": request.baseline_year,
                    "crs": "pixel",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    feature_collection = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(feature_collection, indent=2))
    return {**ctx, "change_path": out_path}
