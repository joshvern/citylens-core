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
    """Partial coverage (modified_iou ≤ IoU < unchanged_iou) → modified."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    # 10x10 baseline; current mask covers only ~30% (IoU ≈ 0.3). Stays
    # comfortably in the modified range for either an unchanged_iou of
    # 0.5 (current default) or 0.6 (historical).
    base[2:12, 2:12] = 1
    img[2:12, 2:5] = 1  # 3 cols of 10 → area ratio ~0.3

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert "modified" in kinds
    modified = [f for f in payload["features"] if f["properties"]["change_type"] == "modified"][0]
    iou = modified["properties"]["baseline_iou"]
    assert 0.2 <= iou < 0.5, f"expected modified-band IoU, got {iou}"
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


def _write_baseline_geojson(tmp_path, features):
    """Helper: materialize a baseline_footprints.geojson the stage can read.

    Uses a pixel-space CRS hint so tests don't have to supply a real transform
    for rasterization. Each feature's geometry must be in pixel coords.
    """
    payload = {
        "type": "FeatureCollection",
        "crs": "pixel",
        "features": features,
    }
    (tmp_path / "baseline_footprints.geojson").write_text(json.dumps(payload))


def test_per_source_feature_splits_adjacent_row_houses(tmp_path: Path, monkeypatch) -> None:
    """Regression: two adjacent row-house footprints that share a wall merge
    into a single component when rasterized, hiding a 2:1 undercount in the
    baseline. The per-source-feature path keeps them separate."""
    from affine import Affine

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")  # ~any size passes
    # Scale=1 means 1 pixel == 1 "m" so px_area_m2 == 1 in this test.

    # Two adjacent row houses, each 6x6, sharing the edge at x=9.
    # Pixel coords (col,row):   A = (3..8, 2..7)     B = (9..14, 2..7)
    # Baseline raster built from these has them touching -> one component.
    feat_a = {
        "type": "Feature",
        "properties": {"Source": "NYC OpenData", "SourceDate": "2017-01-01"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[3, 2], [9, 2], [9, 8], [3, 8], [3, 2]]],
        },
    }
    feat_b = {
        "type": "Feature",
        "properties": {"Source": "NYC OpenData", "SourceDate": "2017-01-01"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[9, 2], [15, 2], [15, 8], [9, 8], [9, 2]]],
        },
    }
    _write_baseline_geojson(tmp_path, [feat_a, feat_b])

    # Baseline pixel mask = union of the two row houses.
    base = np.zeros((20, 20), dtype=np.uint8)
    base[2:8, 3:15] = 1
    # Current-year imagery still has both houses (IoU ≈ 1 for each).
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:8, 3:15] = 1

    payload, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=Affine.identity(), crs="EPSG:3857",
    )

    # Per-source path must be engaged.
    assert summary.qa["change_source"] == "per_source_feature"
    # TWO features, one per row house — NOT one merged component.
    assert len(payload["features"]) == 2
    assert summary.qa["change_counts"]["unchanged"] == 2

    # Provenance is preserved.
    for f in payload["features"]:
        assert f["properties"].get("Source") == "NYC OpenData"
        assert f["properties"].get("SourceDate") == "2017-01-01"
        # Provenance doesn't shadow computed fields.
        assert f["properties"]["change_type"] == "unchanged"


def test_per_source_feature_fallback_when_no_geojson(tmp_path: Path, monkeypatch) -> None:
    """Without baseline_footprints.geojson on disk, stage falls back to the
    component-labeling path and still produces valid output."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[5:15, 5:15] = 1
    img[5:15, 5:15] = 1
    # NOTE: no baseline_footprints.geojson written.

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_source"] == "component_labeled"
    assert summary.qa["change_counts"]["unchanged"] == 1


def test_per_source_feature_handles_multipolygon_geometry(tmp_path: Path, monkeypatch) -> None:
    """MultiPolygon source features emit one output feature per outer ring —
    same shape as the input so Brooklyn row-house-blocks stay faithful."""
    from affine import Affine

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")

    multi = {
        "type": "Feature",
        "properties": {"source_gdb": "Kings_Building_Footprints.gdb"},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[1, 1], [4, 1], [4, 4], [1, 4], [1, 1]]],
                [[[7, 7], [10, 7], [10, 10], [7, 10], [7, 7]]],
            ],
        },
    }
    _write_baseline_geojson(tmp_path, [multi])

    base = np.zeros((15, 15), dtype=np.uint8)
    base[1:4, 1:4] = 1
    base[7:10, 7:10] = 1
    img = base.copy()

    payload, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=Affine.identity(), crs="EPSG:3857",
    )
    # One GDB row = one change event, regardless of how many polygons the
    # MultiPolygon source has. Two Polygon features are emitted for UI
    # convenience (most map libraries handle Polygon better than
    # MultiPolygon), but both carry the same classification + source_gdb.
    assert len(payload["features"]) == 2
    assert summary.qa["change_counts"]["unchanged"] == 1
    assert all(
        f["properties"]["change_type"] == "unchanged"
        for f in payload["features"]
    )
    assert all(
        f["properties"].get("source_gdb") == "Kings_Building_Footprints.gdb"
        for f in payload["features"]
    )


def test_per_source_feature_requires_transform(tmp_path: Path, monkeypatch) -> None:
    """If the caller didn't supply a transform/CRS (pure-pixel-space tests),
    the per-source path is skipped (rasterize needs a transform) and we fall
    back to component-labeling."""
    _write_baseline_geojson(tmp_path, [
        {
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]},
        }
    ])
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:8, 2:8] = 1
    img[2:8, 2:8] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    # No transform -> pixel_space_only -> component-labeled fallback.
    assert summary.qa["change_source"] == "component_labeled"


def test_lidar_height_gate_rejects_short_added_components(
    tmp_path: Path, monkeypatch
) -> None:
    """An 'added' candidate whose LiDAR height is below the threshold is
    rejected. Simulates a tree/shadow/pavement blob SAM2 segmented."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M", "2.0")

    # 5x5 isolated component in imagery, empty baseline → would be 'added'.
    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    # LiDAR heights grid with ALL values at 0.5m above ground. Clearly below
    # the 2m threshold → reject.
    heights = np.full((30, 30), 1.5, dtype=np.float32)
    ground = 1.0  # 0.5m above ground

    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(
        address="x", segmentation_backend="sam2",
        imagery_year=2024, baseline_year=2017,
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc),
    )
    ctx = {
        "mask": img, "baseline_mask": base,
        "lidar_heights": heights, "lidar_ground_z": ground,
    }
    stage_change(req, tmp_path, ctx, summary)
    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["added_rejected"]["too_short"] == 1


def test_lidar_height_gate_accepts_tall_added_components(
    tmp_path: Path, monkeypatch
) -> None:
    """An 'added' candidate whose LiDAR height clears the threshold is kept."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M", "2.0")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    # 10m above ground — a real one-story building.
    heights = np.full((30, 30), 10.0, dtype=np.float32)
    ground = 0.0

    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(
        address="x", segmentation_backend="sam2",
        imagery_year=2024, baseline_year=2017,
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc),
    )
    ctx = {
        "mask": img, "baseline_mask": base,
        "lidar_heights": heights, "lidar_ground_z": ground,
    }
    stage_change(req, tmp_path, ctx, summary)
    assert summary.qa["change_counts"]["added"] == 1
    assert summary.qa["added_rejected"]["too_short"] == 0


def test_lidar_height_gate_skipped_when_lidar_absent(
    tmp_path: Path, monkeypatch
) -> None:
    """Without LiDAR on ctx, the gate doesn't run and components pass
    based on area + overlap filters alone (legacy behavior)."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["added"] == 1
    assert summary.qa["added_rejected"] == {
        "too_small": 0, "baseline_overlap": 0, "too_short": 0,
    }


def test_lidar_height_gate_rejects_when_component_has_no_lidar_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    """Component with no in-bounds LiDAR points is rejected — we can't
    verify it's a building, so err on the side of precision over recall."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M", "2.0")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[20:25, 20:25] = 1   # component in lower-right

    # LiDAR grid is all NaN in the component region.
    heights = np.full((30, 30), np.nan, dtype=np.float32)
    heights[0:5, 0:5] = 10.0   # LiDAR covers a DIFFERENT region
    ground = 0.0

    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(
        address="x", segmentation_backend="sam2",
        imagery_year=2024, baseline_year=2017,
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc),
    )
    ctx = {
        "mask": img, "baseline_mask": base,
        "lidar_heights": heights, "lidar_ground_z": ground,
    }
    stage_change(req, tmp_path, ctx, summary)
    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["added_rejected"]["too_short"] == 1
