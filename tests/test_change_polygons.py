"""Per-building change classification tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from affine import Affine


def _run_stage(tmp_path: Path, *, mask, baseline_mask, transform=None, crs=None):
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(
        address="x",
        segmentation_backend="sam2",
        imagery_year=2024,
        baseline_year=2017,
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    ctx: dict = {"mask": mask, "baseline_mask": baseline_mask}
    if transform is not None:
        ctx["orthophoto_transform"] = transform
    if crs is not None:
        ctx["orthophoto_crs"] = crs
    out = stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())
    return payload, summary, out


def test_unchanged_building_gets_unchanged_feature(tmp_path: Path, monkeypatch) -> None:
    """A baseline footprint well-covered by the current mask (IoU ≥ 0.6) → unchanged."""
    # Disable area filter to keep test geometries small.
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    # 10x10 baseline footprint, current mask covers it almost exactly.
    base[2:12, 2:12] = 1
    img[2:12, 2:12] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds == ["unchanged"]
    assert payload["features"][0]["properties"]["baseline_iou"] >= 0.99
    assert summary.qa["change_counts"] == {
        "unchanged": 1, "modified": 0, "demolished": 0, "added": 0,
    }


def test_modified_building_gets_modified_feature(tmp_path: Path, monkeypatch) -> None:
    """Partial coverage (0.2 ≤ IoU < 0.6) → modified."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    # 10x10 baseline; current mask covers only the left half (IoU ≈ 0.5).
    base[2:12, 2:12] = 1
    img[2:12, 2:7] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert "modified" in kinds
    modified = [f for f in payload["features"] if f["properties"]["change_type"] == "modified"][0]
    assert 0.2 <= modified["properties"]["baseline_iou"] < 0.6
    assert summary.qa["change_counts"]["modified"] == 1


def test_demolished_building_gets_demolished_feature(tmp_path: Path, monkeypatch) -> None:
    """Baseline footprint with ~no current coverage (IoU < 0.2) → demolished."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1
    # current mask is somewhere else entirely

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds == ["demolished"]
    assert payload["features"][0]["properties"]["baseline_iou"] < 0.2
    assert summary.qa["change_counts"]["demolished"] == 1


def test_added_building_gets_added_feature(tmp_path: Path, monkeypatch) -> None:
    """Current-year component that doesn't touch any baseline footprint → added."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "16")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    # Baseline footprint top-left.
    base[1:5, 1:5] = 1
    img[1:5, 1:5] = 1  # matching current-year building → unchanged
    # Isolated 5x5 new building bottom-right (no baseline overlap).
    img[20:25, 20:25] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    added = [f for f in payload["features"] if f["properties"]["change_type"] == "added"]
    assert len(added) == 1
    # 'added' features don't get a baseline_iou.
    assert added[0]["properties"].get("baseline_iou") is None
    assert summary.qa["change_counts"]["added"] == 1


def test_sliver_along_baseline_edge_is_NOT_flagged_as_added(
    tmp_path: Path, monkeypatch
) -> None:
    """Edge-sliver suppression — the big win over the old XOR approach.

    When SAM2 traces a building 2 pixels wider than the baseline footprint,
    the resulting 'added' sliver along the edge should NOT appear as a new
    building feature.
    """
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    # Allow tiny components through the area filter so we can verify the
    # baseline-overlap filter specifically.

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[5:15, 5:15] = 1
    # Current mask is baseline + a 1-pixel-wide strip on the right edge.
    img[5:15, 5:16] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    # Edge sliver touches the baseline → filtered out, no 'added' feature.
    assert summary.qa["change_counts"]["added"] == 0
    # The original baseline building is still classified (as unchanged).
    assert summary.qa["change_counts"]["unchanged"] == 1


def test_min_area_m2_drops_tiny_components(tmp_path: Path, monkeypatch) -> None:
    """In a georeferenced run with 2m x 2m pixels, a 4-pixel added component
    = 16 m² — should be dropped when min area is 50 m², kept at 1 m²."""
    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    # Isolated 2x2 new building (16 m² at 2m/px).
    img[20:22, 20:22] = 1
    transform = Affine.translation(0, 0) * Affine.scale(2, -2)

    # Strict floor — drops the component.
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "50")
    payload, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )
    assert summary.qa["change_counts"]["added"] == 0

    # Permissive floor — keeps it.
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "1")
    payload, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )
    assert summary.qa["change_counts"]["added"] == 1


def test_iou_thresholds_are_env_tunable(tmp_path: Path, monkeypatch) -> None:
    """Bump the unchanged threshold so a partial building is reclassified
    as 'modified' instead of 'unchanged'."""
    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    # 10x10 baseline; current covers all but 1 row (IoU ≈ 0.9).
    base[2:12, 2:12] = 1
    img[2:11, 2:12] = 1

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    # Default threshold: this is 'unchanged' (IoU ~ 0.9 >= 0.6).
    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["unchanged"] == 1

    # Raise the bar so IoU 0.9 is no longer "unchanged" → falls to modified.
    monkeypatch.setenv("CITYLENS_CHANGE_UNCHANGED_IOU", "0.95")
    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["unchanged"] == 0
    assert summary.qa["change_counts"]["modified"] == 1


def test_feature_schema_has_all_expected_fields(tmp_path: Path, monkeypatch) -> None:
    """Regression: every feature must carry the full set of properties the
    frontend expects."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1
    img[2:12, 2:12] = 1

    payload, _, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=Affine.identity(), crs="EPSG:3857",
    )
    assert len(payload["features"]) == 1
    props = payload["features"][0]["properties"]
    for k in ("change_type", "imagery_year", "baseline_year", "crs"):
        assert k in props, f"missing {k} in {props}"
    assert props["imagery_year"] == 2024
    assert props["baseline_year"] == 2017
    assert "baseline_iou" in props  # set for unchanged/modified/demolished
