from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from affine import Affine


def test_two_separate_added_components_emit_two_features(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    # Two disjoint 4x4 imagery blocks, baseline empty.
    imagery = np.zeros((20, 20), dtype=np.uint8)
    imagery[1:5, 1:5] = 1     # top-left block
    imagery[12:16, 12:16] = 1  # bottom-right block

    ctx = {
        "mask": imagery,
        "baseline_mask": np.zeros((20, 20), dtype=np.uint8),
    }

    stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())

    kinds = [f["properties"]["kind"] for f in payload["features"]]
    assert kinds.count("added") == 2
    assert "removed" not in kinds
    for f in payload["features"]:
        assert f["geometry"]["type"] == "Polygon"
        assert len(f["geometry"]["coordinates"][0]) >= 4


def test_tiny_speck_components_are_dropped(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    # One real 4x4 component (16 px) + two 1-px specks. Specks fall below
    # the default 8-px noise floor and should be silently dropped.
    imagery = np.zeros((20, 20), dtype=np.uint8)
    imagery[0:4, 0:4] = 1
    imagery[10, 10] = 1
    imagery[15, 18] = 1

    ctx = {
        "mask": imagery,
        "baseline_mask": np.zeros((20, 20), dtype=np.uint8),
    }

    stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())

    assert len(payload["features"]) == 1


def test_added_and_removed_both_get_per_component_polygons(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    imagery = np.zeros((30, 30), dtype=np.uint8)
    baseline = np.zeros((30, 30), dtype=np.uint8)
    # Two new buildings in the imagery (added).
    imagery[2:8, 2:8] = 1
    imagery[2:8, 20:26] = 1
    # One demolished building in the baseline (removed).
    baseline[20:26, 10:16] = 1

    ctx = {"mask": imagery, "baseline_mask": baseline}
    stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())

    kinds = [f["properties"]["kind"] for f in payload["features"]]
    assert kinds.count("added") == 2
    assert kinds.count("removed") == 1


def test_env_var_tunes_min_area(tmp_path: Path, monkeypatch) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    # One 4-pixel component. Default threshold (8) drops it; overriding to 1
    # keeps it.
    imagery = np.zeros((10, 10), dtype=np.uint8)
    imagery[0:2, 0:2] = 1
    ctx = {"mask": imagery, "baseline_mask": np.zeros((10, 10), dtype=np.uint8)}

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())
    assert len(payload["features"]) == 1

    # And with a very aggressive threshold (100), it drops even larger features.
    imagery2 = np.zeros((10, 10), dtype=np.uint8)
    imagery2[0:5, 0:5] = 1  # 25 px
    ctx2 = {"mask": imagery2, "baseline_mask": np.zeros((10, 10), dtype=np.uint8)}
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "100")
    stage_change(req, tmp_path, ctx2, summary)
    payload2 = json.loads((tmp_path / "change.geojson").read_text())
    assert len(payload2["features"]) == 0
