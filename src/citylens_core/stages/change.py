from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


def stage_change(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "change.geojson"

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "kind": "placeholder",
                    "imagery_year": request.imagery_year,
                    "baseline_year": request.baseline_year,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-73.9857, 40.7484],
                            [-73.9856, 40.7484],
                            [-73.9856, 40.7485],
                            [-73.9857, 40.7485],
                            [-73.9857, 40.7484],
                        ]
                    ],
                },
            }
        ],
    }

    out_path.write_text(json.dumps(feature_collection, indent=2))
    return {**ctx, "change_path": out_path}
