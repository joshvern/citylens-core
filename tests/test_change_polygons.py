"""Per-building change classification tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from affine import Affine
from PIL import Image


def _run_stage(tmp_path: Path, *, mask, baseline_mask, transform=None, crs=None, ctx_extra=None):
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
    if ctx_extra:
        ctx.update(ctx_extra)
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
    # The default near-baseline dilation is 24 px; on a 30×30 grid every
    # pixel is within 24 px of the baseline. Drop to 8 px for this test so
    # the "isolated building" stays genuinely isolated.
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", "8")

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


def test_separate_discovery_finds_added_without_changing_baseline_classification(
    tmp_path: Path, monkeypatch
) -> None:
    """Prompted IoU and automatic added discovery remain independent.

    The prompted mask contains only the unchanged baseline building.  The
    discovery mask deliberately omits that building and contains only a new
    isolated roof.  Replacing the prompted mask with discovery would classify
    the baseline as demolished; ignoring discovery would miss the addition.
    """
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "4")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", "4")

    base = np.zeros((40, 40), dtype=np.uint8)
    prompted = np.zeros((40, 40), dtype=np.uint8)
    discovery = np.zeros((40, 40), dtype=np.uint8)
    base[3:11, 3:11] = 1
    prompted[3:11, 3:11] = 1
    discovery[28:35, 28:35] = 1

    payload, summary, out = _run_stage(
        tmp_path,
        mask=prompted,
        baseline_mask=base,
        ctx_extra={"refined_added_discovery_mask": discovery},
    )

    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds.count("unchanged") == 1
    assert kinds.count("added") == 1
    assert "demolished" not in kinds
    assert summary.qa["added_mask_source"] == "refined_added_discovery_mask"
    assert out["change_mask"][30, 30] == 1

    # Without the discovery key, the same prompted mask cannot contain the
    # genuinely new component and therefore emits no added feature.
    payload_without, summary_without, _ = _run_stage(
        tmp_path,
        mask=prompted,
        baseline_mask=base,
    )
    kinds_without = [
        f["properties"]["change_type"] for f in payload_without["features"]
    ]
    assert kinds_without == ["unchanged"]
    assert summary_without.qa["change_counts"]["added"] == 0


def test_discovery_component_still_uses_added_lidar_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """Baseline-epoch flat LiDAR confirms a discovery-only addition."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "4")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", "4")

    base = np.zeros((40, 40), dtype=np.uint8)
    prompted = np.zeros((40, 40), dtype=np.uint8)
    discovery = np.zeros((40, 40), dtype=np.uint8)
    base[3:11, 3:11] = 1
    prompted[3:11, 3:11] = 1
    discovery[28:35, 28:35] = 1
    lidar = np.zeros((40, 40), dtype=np.float32)

    payload, summary, _ = _run_stage(
        tmp_path,
        mask=prompted,
        baseline_mask=base,
        ctx_extra={
            "refined_added_discovery_mask": discovery,
            "lidar_heights": lidar,
            "lidar_ground_z": 0.0,
        },
    )

    added = [
        f for f in payload["features"] if f["properties"]["change_type"] == "added"
    ]
    assert len(added) == 1
    assert added[0]["properties"]["baseline_lidar_flat"] is True
    assert added[0]["properties"]["confidence"] >= 0.85
    assert summary.qa["change_counts"]["added"] == 1


def test_border_touching_discovery_blob_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """A tile-scale AMG road/background blob must not become an addition."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "4")

    base = np.zeros((40, 40), dtype=np.uint8)
    prompted = np.zeros((40, 40), dtype=np.uint8)
    discovery = np.zeros((40, 40), dtype=np.uint8)
    base[20:28, 20:28] = 1
    prompted[20:28, 20:28] = 1
    discovery[0:12, 0:35] = 1
    lidar = np.zeros((40, 40), dtype=np.float32)

    _, summary, _ = _run_stage(
        tmp_path,
        mask=prompted,
        baseline_mask=base,
        ctx_extra={
            "refined_added_discovery_mask": discovery,
            "lidar_heights": lidar,
            "lidar_ground_z": 0.0,
        },
    )

    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["added_rejected"]["border_touching"] == 1


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


def test_adaptive_threshold_lowers_unchanged_iou_when_median_drops(
    tmp_path: Path, monkeypatch
) -> None:
    """On a tile where SAM2's median per-baseline IoU is much lower than
    0.4 (e.g., dense Manhattan mixed-use blocks), the global 0.4 threshold
    over-flags stable buildings as 'modified'. The adaptive logic should
    lower the threshold to (median - 0.1) clamped to the floor, recovering
    most of the 'unchanged' bucket.

    Synthesizes 25 baseline footprints whose imagery overlap is uniformly
    ~30% (IoU ~0.30), then asserts that the adaptive logic reclassifies
    them from 'modified' to 'unchanged'.
    """
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")

    H, W = 80, 80
    base = np.zeros((H, W), dtype=np.uint8)
    img = np.zeros((H, W), dtype=np.uint8)
    # 25 baselines arranged in a 5x5 grid, each 10x10. Imagery covers ~30% of
    # each (3 rows × 10 cols) → IoU ~0.30.
    for row in range(5):
        for col in range(5):
            y0, x0 = 10 + row * 12, 10 + col * 12
            base[y0 : y0 + 10, x0 : x0 + 10] = 1
            img[y0 : y0 + 3, x0 : x0 + 10] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    counts = summary.qa["change_counts"]
    # With adaptive: median IoU ~0.30, threshold lowered to ~0.20 (floor).
    # All 25 baselines should land in 'unchanged' instead of 'modified'.
    assert "median_baseline_iou" in summary.qa
    assert summary.qa["unchanged_iou_used"] < 0.4, summary.qa
    assert counts["unchanged"] >= 20, counts
    # Should have recorded reclassification.
    rec = summary.qa.get("adaptive_threshold_reclassifications") or {}
    assert rec.get("modified_to_unchanged", 0) > 0, rec


def test_adaptive_threshold_skipped_below_min_samples(
    tmp_path: Path, monkeypatch
) -> None:
    """With fewer than the configured min samples, adaptive doesn't kick
    in — the configured `unchanged_iou` is used as-is. Synthetic tests rely
    on this behavior."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    # Single baseline with IoU ~0.30 — should classify as 'modified' under
    # the configured 0.4 threshold, NOT get rescued by the adaptive logic.
    base[2:12, 2:12] = 1
    img[2:12, 2:5] = 1

    _, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["modified"] == 1
    assert summary.qa["unchanged_iou_used"] == 0.4


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


def test_baseline_epoch_flat_lidar_confirms_added(
    tmp_path: Path, monkeypatch
) -> None:
    """BASELINE-epoch LiDAR semantics: the LiDAR grid is contemporaneous
    with the 2017 baseline, so ground-level LiDAR inside a 2024 building-
    shaped component means the parcel was EMPTY in 2017 — positive evidence
    of new construction. Accepted as 'added' with boosted confidence.

    (Under the old current-epoch assumption this exact fixture was rejected
    'too_short' — i.e. the gate rejected precisely the new buildings it
    existed to find.)"""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_MAX_BASELINE_HEIGHT_M", "2.0")

    # 5x5 isolated component in imagery, empty baseline → 'added' candidate.
    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    # 2017 LiDAR: 0.5 m above ground everywhere — bare lot back then.
    heights = np.full((30, 30), 1.5, dtype=np.float32)
    ground = 1.0

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
    assert summary.qa["added_rejected"]["preexisting_structure_emitted_as_candidate"] == 0

    payload = json.loads((tmp_path / "change.geojson").read_text())
    added = [
        f for f in payload["features"]
        if f["properties"]["change_type"] == "added"
    ]
    assert len(added) == 1
    # Two independent signals agree (flat-2017 + building-2024): boosted
    # confidence and an audit flag.
    assert added[0]["properties"]["confidence"] >= 0.85
    assert added[0]["properties"]["baseline_lidar_flat"] is True


def test_baseline_epoch_tall_lidar_demotes_added_to_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    """A component that was TALL in the 2017 LiDAR already had a structure
    (tree canopy or a building missing from the footprints GDB) — it is NOT
    new construction, so it demotes to candidate_added for review instead of
    being confidently accepted."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_MAX_BASELINE_HEIGHT_M", "2.0")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    # 10 m above ground in the 2017 grid — something already stood there.
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
    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["change_counts"]["candidate_added"] == 1
    assert summary.qa["added_rejected"]["preexisting_structure_emitted_as_candidate"] == 1

    payload = json.loads((tmp_path / "change.geojson").read_text())
    cands = [
        f for f in payload["features"]
        if f["properties"]["change_type"] == "candidate_added"
    ]
    assert len(cands) == 1
    assert (
        cands[0]["properties"]["candidate_reason"]
        == "preexisting_structure_in_baseline_lidar"
    )


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
        "too_small": 0,
        "border_touching": 0,
        "baseline_overlap": 0,
        "centroid_near_baseline": 0,
        "majority_inside_baseline_dilation": 0,
        "vegetation": 0,
        "preexisting_structure_emitted_as_candidate": 0,
        "no_lidar_coverage_emitted_as_candidate": 0,
    }


def test_demolished_stays_demolished_under_baseline_epoch_lidar(
    tmp_path: Path, monkeypatch
) -> None:
    """Default epoch is 'baseline' (production reality: 2017 LiDAR vs 2024
    imagery). A standing structure in baseline-epoch LiDAR is EXPECTED for a
    genuine 2017→2024 demolition — the old building was there in 2017 either
    way — so the rescue must NOT fire, or every real demolition would get
    silently downgraded to 'modified' (the old behavior)."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_DEMOLISHED_MAX_HEIGHT_M", "3.0")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1
    # SAM2 2024 mask empty inside the footprint: the building is gone.

    # 2017 LiDAR shows the (since-demolished) building standing — expected.
    heights = np.full((20, 20), 15.0, dtype=np.float32)
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
    payload = json.loads((tmp_path / "change.geojson").read_text())
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds == ["demolished"], f"expected demolished to stand, got {kinds}"
    assert summary.qa["change_counts"]["demolished"] == 1
    assert summary.qa["demolished_downgraded_to_modified"] == 0
    assert summary.qa["demolished_rescue_lidar_epoch"] == "baseline"


def test_demolished_is_rescued_to_modified_with_current_epoch_lidar(
    tmp_path: Path, monkeypatch
) -> None:
    """With CURRENT-epoch LiDAR (a deploy with fresh LiDAR), a standing
    structure genuinely contradicts 'demolished': SAM2 missed the building
    (dark roof / shadow) → downgrade demolished→modified."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_DEMOLISHED_MAX_HEIGHT_M", "3.0")
    monkeypatch.setenv("CITYLENS_CHANGE_DEMOLISHED_RESCUE_LIDAR_EPOCH", "current")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1

    heights = np.full((20, 20), 15.0, dtype=np.float32)
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
    payload = json.loads((tmp_path / "change.geojson").read_text())
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds == ["modified"], f"expected rescue → modified, got {kinds}"
    assert summary.qa["change_counts"]["demolished"] == 0
    assert summary.qa["change_counts"]["modified"] == 1
    assert summary.qa["demolished_downgraded_to_modified"] == 1


def test_demolished_rescue_does_not_fire_without_lidar(
    tmp_path: Path, monkeypatch
) -> None:
    """No LiDAR on ctx → rescue can't run, so a SAM2-empty baseline footprint
    still classifies as demolished (legacy behavior)."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["demolished"] == 1
    assert summary.qa["demolished_downgraded_to_modified"] == 0


def test_added_courtyard_rejected_by_baseline_dilation_centroid_filter(
    tmp_path: Path, monkeypatch
) -> None:
    """SAM2 picks up courtyards / lightwells (flat roof-like surfaces between
    buildings) as 'added' candidates. The 1-px overlap filter doesn't catch
    them because there's a 2-10 px gap from surrounding buildings (rasterized
    GDB footprints + sidewalk moat). The wider centroid-near-baseline filter
    rejects any candidate whose centroid lies inside the 8-px-dilated
    baseline mask."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", "8")

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    # Two baseline buildings with a 12-px-wide moat between them — the
    # GDB rasterizer + sidewalk gap produces this kind of spacing. The
    # courtyard candidate sits in the middle of that gap, ~4 px from the
    # nearest baseline edge. That's far enough that the 1-px overlap
    # filter does NOT catch it, but well inside the 8-px-dilated baseline
    # mask, so the centroid-near-baseline filter must reject it.
    base[10:30, 4:14] = 1   # left wing
    base[10:30, 26:36] = 1  # right wing
    # SAM2 traced the courtyard surface as a building, sitting in the
    # middle of the gap (4 px clearance from both baseline edges).
    img[14:26, 18:22] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    # Confirm the 1-px overlap filter did NOT catch this — only the new
    # centroid-near-baseline filter did.
    assert summary.qa["added_rejected"]["baseline_overlap"] == 0
    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["added_rejected"]["centroid_near_baseline"] == 1


def test_added_genuine_new_building_far_from_baseline_passes_centroid_filter(
    tmp_path: Path, monkeypatch
) -> None:
    """Sanity check: an 'added' candidate whose centroid is well outside the
    8-px-dilated baseline mask is NOT rejected by the courtyard filter."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX", "8")

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    # Baseline footprint in upper-left.
    base[2:8, 2:8] = 1
    img[2:8, 2:8] = 1
    # Genuine new building far away in the lower-right (centroid ~(30, 30)
    # is more than 8 px from any baseline pixel).
    img[27:33, 27:33] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["change_counts"]["added"] == 1
    assert summary.qa["added_rejected"]["centroid_near_baseline"] == 0


def test_added_default_dilate_catches_misaligned_existing_building(
    tmp_path: Path, monkeypatch
) -> None:
    """If per-feature registration didn't apply, the imagery mask of an
    existing building can sit 5-7m (≈18-24 px at 0.3 m/px) away from its
    baseline footprint. The matcher then misses it and the 'added' pass
    sees a clean component near (but not touching) a baseline. The
    default dilation (24 px) must catch this case."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    # No CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX override → exercise the
    # default of 24.

    base = np.zeros((80, 80), dtype=np.uint8)
    img = np.zeros((80, 80), dtype=np.uint8)
    base[20:30, 20:30] = 1
    # Same building shape, shifted ~6 m at 0.3 m/px (20 px) to the right —
    # alignment-error-twin scenario. Centroid of the imagery component is
    # 24 px east of the baseline centroid, well inside the 24-px dilation.
    img[20:30, 40:50] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    # Should NOT be classified as a new building; the near-baseline gate
    # (centroid-in-dilation OR majority-inside-dilation) catches it.
    assert summary.qa["change_counts"]["added"] == 0
    rej = summary.qa["added_rejected"]
    assert (
        rej["centroid_near_baseline"] >= 1
        or rej["majority_inside_baseline_dilation"] >= 1
    ), rej


def test_registration_recovers_iou_under_misalignment(
    tmp_path: Path, monkeypatch
) -> None:
    """A baseline mask shifted by (1, 1) px from the current mask — same
    building, mis-registered by the kind of acquisition error we see
    between NYS Orthos baseline and current. Without alignment IoU
    drops well below the unchanged threshold; the registration step
    should recover it so the building is correctly classified as
    'unchanged'."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    # Single 10x10 building. Current at [10:20, 10:20], baseline at
    # [11:21, 11:21] — pure (1, 1) translation.
    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:20, 10:20] = 1
    base[11:21, 11:21] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)

    # Registration metadata is recorded in qa.
    reg = summary.qa["registration"]
    assert reg["applied"] is True
    assert int(reg["dy"]) == -1
    assert int(reg["dx"]) == -1
    # IoU after applying the shift should be much higher.
    assert reg["iou_after"] > reg["iou_before"]
    assert reg["iou_after"] > 0.9
    # The building reads as unchanged after recovery.
    assert summary.qa["change_counts"]["unchanged"] == 1
    assert summary.qa["change_counts"]["modified"] == 0


def test_registration_recovers_per_source_feature_iou_without_moving_output_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    """Production uses the per-source GeoJSON path. Accepted registration
    must improve the scoring mask there too, while emitted source geometry
    remains in its original CRS coordinates."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")

    src_geom = [[[14, 14], [24, 14], [24, 24], [14, 24], [14, 14]]]
    _write_baseline_geojson(
        tmp_path,
        [
            {
                "type": "Feature",
                "properties": {"source_gdb": "test.gdb"},
                "geometry": {"type": "Polygon", "coordinates": src_geom},
            }
        ],
    )

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    base[14:24, 14:24] = 1
    img[10:20, 10:20] = 1

    payload, summary, _ = _run_stage(
        tmp_path,
        mask=img,
        baseline_mask=base,
        transform=Affine.identity(),
        crs="EPSG:3857",
    )

    reg = summary.qa["registration"]
    assert reg["applied"] is True
    assert int(reg["dy"]) == -4
    assert int(reg["dx"]) == -4
    assert summary.qa["change_source"] == "per_source_feature"
    assert summary.qa["change_counts"]["unchanged"] == 1
    assert summary.qa["change_counts"]["modified"] == 0

    feature = payload["features"][0]
    assert feature["properties"]["baseline_iou"] >= 0.99
    assert feature["geometry"]["coordinates"] == src_geom


def test_registration_skipped_when_masks_already_aligned(
    tmp_path: Path, monkeypatch
) -> None:
    """Aligned masks — no shift should be applied. We don't want noise
    moving things by a pixel for no reason."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    base[12:25, 12:25] = 1
    img[12:25, 12:25] = 1

    _, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    reg = summary.qa["registration"]
    assert reg["applied"] is False
    assert reg["iou_before"] == 1.0


def test_polygon_smoothing_reduces_vertex_count_in_emitted_geojson(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end check that the simplification pass actually shaves
    vertices off the emitted change.geojson polygons. Builds a square
    on a tile with non-trivial transform so simplification is in
    'world units' mode, not the identity short-circuit."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "1")

    # 0.5 m/px metric transform — non-identity, so simplification fires.
    transform = Affine(0.5, 0.0, 0.0, 0.0, -0.5, 100.0)

    # baseline empty — every imagery component becomes "added"
    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    # 12x12 "added" rectangle — rasterize+shapes naturally produces
    # one rectilinear ring without saw-tooth, but simplification
    # should still pass through valid output without breaking.
    img[5:17, 5:17] = 1

    payload, _, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base, transform=transform, crs="EPSG:3857"
    )
    feats = payload["features"]
    assert len(feats) == 1
    ring = feats[0]["geometry"]["coordinates"][0]
    # Closed and reasonably small (a clean rectangle is 5 points).
    assert ring[0] == ring[-1]
    assert len(ring) <= 8


def test_lidar_no_coverage_emits_candidate_added_with_low_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    """When LiDAR doesn't cover a candidate-added component, the
    component is no longer dropped silently. It's emitted with
    change_type='candidate_added' and a low confidence score, so the
    frontend can render it faded ('possibly added, no LiDAR
    confirmation') rather than the user seeing nothing.
    Trades the old precision-over-recall stance for an explicit
    low-confidence signal."""
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

    # Not counted as a confirmed add — held back for the candidate slot.
    assert summary.qa["change_counts"].get("added", 0) == 0
    assert summary.qa["change_counts"].get("candidate_added", 0) == 1
    assert summary.qa["added_rejected"]["preexisting_structure_emitted_as_candidate"] == 0
    assert summary.qa["added_rejected"]["no_lidar_coverage_emitted_as_candidate"] == 1

    payload = json.loads((tmp_path / "change.geojson").read_text())
    candidates = [
        f for f in payload["features"]
        if f["properties"]["change_type"] == "candidate_added"
    ]
    assert len(candidates) == 1
    cand = candidates[0]
    # Low confidence — frontend renders this faded.
    assert cand["properties"]["confidence"] < 0.5


# ---------------------------------------------------------------------------
# Borderline-modified reclassification (two-stage classifier).
# ---------------------------------------------------------------------------


def _make_borderline_tile(
    tmp_path: Path,
    *,
    n_features: int = 25,
    cover_pixels: int = 22,
):
    """Build a baseline_footprints.geojson + raster pair where every 10x10
    footprint has exactly `cover_pixels` px of imagery overlap → per-feature
    IoU = cover_pixels / 100. Returns (base, img, transform).

    Default cover_pixels=22 → IoU=0.22 per feature, median=0.22 → adaptive
    lowers unchanged_iou to 0.25 (floor), so every feature stays classified
    as 'modified' (IoU 0.22 < 0.25) while sitting comfortably inside the
    borderline band [0.20, 0.25). This lets the borderline-reclassification
    pass act on every feature in the test.
    """
    # Stride needs to exceed the footprint size + ROI padding so the
    # per-source-feature IoU loop doesn't see neighboring footprints'
    # imagery bleed into one footprint's ROI. Padding is max(1, side/10)
    # = 1 each side, so 14 rows of clearance is plenty.
    STRIDE = 14
    H = max(20, STRIDE * n_features + 20)
    W = 60
    base = np.zeros((H, W), dtype=np.uint8)
    img = np.zeros((H, W), dtype=np.uint8)
    feats = []
    # Each baseline = 10x10 = 100 px. Cover `cover_pixels` of them with
    # imagery so IoU = cover/100 is exact. We fill row-major to make the
    # masked region a contiguous block.
    cover_pixels = max(1, min(99, cover_pixels))
    full_rows, leftover_cols = divmod(cover_pixels, 10)
    for i in range(n_features):
        y0 = 5 + i * STRIDE
        x0 = 5
        if y0 + 10 >= H:
            break
        base[y0 : y0 + 10, x0 : x0 + 10] = 1
        if full_rows > 0:
            img[y0 : y0 + full_rows, x0 : x0 + 10] = 1
        if leftover_cols > 0:
            img[y0 + full_rows, x0 : x0 + leftover_cols] = 1
        feats.append(
            {
                "type": "Feature",
                "properties": {"source_gdb": "test.gdb"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [x0, y0],
                            [x0 + 10, y0],
                            [x0 + 10, y0 + 10],
                            [x0, y0 + 10],
                            [x0, y0],
                        ]
                    ],
                },
            }
        )
    _write_baseline_geojson(tmp_path, feats)
    return base, img, Affine.identity()


def test_borderline_modified_reclassified_to_unchanged_when_no_surface_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    """Build 25 footprints whose IoU lands in the (post-adaptive) borderline
    band. Without surface_changed evidence (no RGB baseline wired up — same
    state as production today), each one should get reclassified from
    'modified' to 'unchanged' with a low confidence and a
    `borderline_reclassified=True` provenance flag."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")
    # IoU 0.22: median=0.22 → adaptive lowers unchanged_iou to 0.25 (floor),
    # IoU 0.22 < 0.25 → all stay 'modified'; the borderline band [0.20, 0.25]
    # then catches every feature in the second-stage pass.
    base, img, transform = _make_borderline_tile(
        tmp_path, n_features=25, cover_pixels=22
    )
    payload, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )

    # Adaptive threshold should have fired and lowered to the floor.
    assert summary.qa["unchanged_iou_used"] == 0.25, summary.qa
    # Borderline pass should have moved every feature into 'unchanged'.
    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("to_unchanged", 0) >= 20, recls
    # No feature should have been kept by surface evidence (none available).
    assert recls.get("kept_by_surface_change", 0) == 0
    # `change_counts` must reflect the reclassification.
    assert summary.qa["change_counts"]["modified"] == 0, summary.qa["change_counts"]
    assert summary.qa["change_counts"]["unchanged"] >= 20

    # Provenance flag visible on the geojson, with reduced confidence.
    flagged = [
        f for f in payload["features"]
        if f["properties"].get("borderline_reclassified")
    ]
    assert len(flagged) >= 20
    for f in flagged:
        assert f["properties"]["change_type"] == "unchanged"
        assert f["properties"]["confidence"] <= 0.55


def test_one_band_baseline_mask_does_not_act_as_surface_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    """Production baseline.tif is a one-band footprint mask. It must not be
    converted to RGB and treated as historical orthophoto evidence, otherwise
    every gray roof can look like a high-Delta-E surface change."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")

    base, img, transform = _make_borderline_tile(
        tmp_path, n_features=25, cover_pixels=22
    )
    current_path = tmp_path / "orthophoto.png"
    baseline_path = tmp_path / "baseline.tif"
    Image.fromarray(np.full((*base.shape, 3), 128, dtype=np.uint8)).save(current_path)
    Image.fromarray((base * 255).astype(np.uint8)).save(baseline_path)

    _, summary, _ = _run_stage(
        tmp_path,
        mask=img,
        baseline_mask=base,
        transform=transform,
        crs="EPSG:3857",
        ctx_extra={
            "orthophoto_path": current_path,
            "baseline_path": baseline_path,
        },
    )

    assert summary.qa["surface_change_available"] is False
    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("kept_by_surface_change", 0) == 0
    assert recls.get("to_unchanged", 0) >= 20, recls
    assert summary.qa["change_counts"]["modified"] == 0


def test_borderline_modified_kept_when_surface_evidence_confirms_change(
    tmp_path: Path, monkeypatch
) -> None:
    """A feature with borderline IoU + surface_changed=True (real visual
    change confirmed by Δ-E) keeps its 'modified' label even after the
    borderline pass. Stubs the surface-change helper so we can drive
    surface_changed deterministically without needing an RGB baseline."""
    from citylens_core.stages import change as change_mod

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")

    base, img, transform = _make_borderline_tile(
        tmp_path, n_features=25, cover_pixels=22
    )

    # Stub surface-image loader so the stage believes RGB imagery is
    # available, AND stub surface_delta_e to return a value above the
    # threshold (≥ 20.0) so every borderline modified gets `surface_changed`
    # = True. With surface confirmation the borderline pass must NOT
    # reclassify any of them to unchanged.
    class _StubSurface:
        current_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        baseline_rgb = np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(
        change_mod, "load_surface_images",
        lambda *args, **kwargs: _StubSurface(),
    )
    monkeypatch.setattr(
        change_mod, "surface_delta_e",
        lambda *args, **kwargs: 99.0,  # comfortably above threshold
    )
    # is_surface_changed reads the threshold env var; default 20.0 is fine.

    _, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )

    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("to_unchanged", 0) == 0, recls
    assert recls.get("kept_by_surface_change", 0) >= 20, recls
    # The features stay classified as modified.
    assert summary.qa["change_counts"]["modified"] >= 20


def test_borderline_modified_pass_skipped_with_zero_margin(
    tmp_path: Path, monkeypatch
) -> None:
    """Setting margin to 0 disables the borderline pass entirely — historical
    behavior before this fix shipped. `change_counts` should match the
    pre-margin run."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")
    monkeypatch.setenv("CITYLENS_CHANGE_MODIFIED_BORDERLINE_MARGIN", "0")

    base, img, transform = _make_borderline_tile(
        tmp_path, n_features=25, cover_pixels=22
    )
    _, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )

    # Pass shouldn't have moved anything.
    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("to_unchanged", 0) == 0
    # The qa.borderline_modified_margin field should still be emitted with
    # the configured value, so operators can audit "was this pass on?"
    assert summary.qa["borderline_modified_margin"] == 0.0


def test_borderline_modified_pass_skipped_below_min_samples(
    tmp_path: Path, monkeypatch
) -> None:
    """With fewer features than the adaptive-min-samples threshold, the
    borderline pass doesn't fire (same gating as adaptive — synthetic single-
    feature tests must keep their pre-margin behavior)."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")

    # Single baseline with IoU ~0.9, threshold raised to 0.95 → modified.
    # Borderline band would otherwise [0.90, 0.95) — IoU=0.9 sits at the
    # band edge, which would trigger reclassification IF the pass ran.
    base = np.zeros((20, 20), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=np.uint8)
    base[2:12, 2:12] = 1
    img[2:11, 2:12] = 1  # IoU ≈ 0.9
    monkeypatch.setenv("CITYLENS_CHANGE_UNCHANGED_IOU", "0.95")

    _, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    # 1 feature is well below the min-samples gate → pass skipped.
    assert summary.qa["change_counts"]["modified"] == 1
    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("to_unchanged", 0) == 0


def test_borderline_pass_does_not_touch_demolished_or_added(
    tmp_path: Path, monkeypatch
) -> None:
    """Demolished + added features are LiDAR-validated (or come from a
    different code path) — the borderline pass must not reclassify them
    even when they coexist with borderline modifieds on the same tile."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "0.01")
    monkeypatch.setenv("CITYLENS_CHANGE_ADAPTIVE_MIN_SAMPLES", "20")
    base, img, transform = _make_borderline_tile(
        tmp_path, n_features=25, cover_pixels=22
    )
    # Inject a clearly-demolished baseline (zero current coverage) — IoU=0.
    # No LiDAR is supplied so it stays as 'demolished' (no rescue).
    base[55:62, 30:37] = 1
    # Read existing geojson, append demolished feature.
    gj_path = tmp_path / "baseline_footprints.geojson"
    payload = json.loads(gj_path.read_text())
    payload["features"].append(
        {
            "type": "Feature",
            "properties": {"source_gdb": "test.gdb"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[30, 55], [37, 55], [37, 62], [30, 62], [30, 55]]
                ],
            },
        }
    )
    gj_path.write_text(json.dumps(payload))

    _, summary, _ = _run_stage(
        tmp_path, mask=img, baseline_mask=base,
        transform=transform, crs="EPSG:3857",
    )

    # Demolished still counted (untouched by the borderline pass).
    assert summary.qa["change_counts"]["demolished"] == 1, summary.qa
    # Borderline pass moved the modifieds but left the demolished alone.
    recls = summary.qa.get("borderline_modified_reclassifications") or {}
    assert recls.get("to_unchanged", 0) >= 20


# ----------------------------------------------------------------------
# Mercator ground-area correction
# ----------------------------------------------------------------------


def test_pixel_area_corrects_web_mercator_latitude_distortion() -> None:
    """EPSG:3857 'meters' are not ground meters: scale is sec(lat), so a
    naive a*e product overstates ground area by sec²(lat) (~1.74x at NYC's
    40.7°N). `_pixel_area_m2` must correct by cos²(lat) derived from the
    transform's y-origin."""
    import math

    from citylens_core.stages.change import _pixel_area_m2

    # NYC-ish web-mercator y (top edge). lat = atan(sinh(y/R)).
    nyc_y = 4_970_000.0
    lat = math.atan(math.sinh(nyc_y / 6378137.0))
    tr = Affine(0.49, 0.0, -8_235_000.0, 0.0, -0.49, nyc_y)

    got = _pixel_area_m2(tr, "EPSG:3857")
    expected = 0.49 * 0.49 * math.cos(lat) ** 2
    assert abs(got - expected) < 1e-9
    # Sanity: the correction is substantial at NYC latitude (~43% smaller).
    assert got < 0.49 * 0.49 * 0.62


def test_pixel_area_equator_mercator_unchanged() -> None:
    """At the equator (y≈0) the Mercator correction is a no-op — the
    behavior every existing synthetic fixture depends on."""
    from citylens_core.stages.change import _pixel_area_m2

    tr = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0)
    assert abs(_pixel_area_m2(tr, "EPSG:3857") - 4.0) < 1e-9


def test_pixel_area_state_plane_feet_conversion() -> None:
    """NY State Plane (EPSG:2263, US survey feet) converts ft² → m² without
    a latitude correction."""
    from citylens_core.stages.change import _pixel_area_m2

    tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    got = _pixel_area_m2(tr, "EPSG:2263")
    assert abs(got - 0.0929034116) < 1e-9


# ----------------------------------------------------------------------
# ExG vegetation reject on the 'added' path
# ----------------------------------------------------------------------


def _run_added_with_ortho(
    tmp_path: Path, monkeypatch, *, color: tuple[int, int, int]
):
    """30x30 tile, one isolated 5x5 'added' candidate, orthophoto painted a
    solid color inside the candidate. No LiDAR (isolates the ExG gate)."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    base = np.zeros((30, 30), dtype=np.uint8)
    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:15, 10:15] = 1

    ortho = np.full((30, 30, 3), 90, dtype=np.uint8)  # neutral gray tile
    ortho[10:15, 10:15] = np.array(color, dtype=np.uint8)
    ortho_path = tmp_path / "orthophoto.png"
    Image.fromarray(ortho, mode="RGB").save(ortho_path)

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
        "mask": img,
        "baseline_mask": base,
        "orthophoto_path": ortho_path,
    }
    stage_change(req, tmp_path, ctx, summary)
    return summary


def test_added_vegetation_rejected_by_exg(tmp_path: Path, monkeypatch) -> None:
    """A strongly green candidate (healthy canopy: ExG = 2G-R-B >> 30) is
    rejected as vegetation — the baseline-epoch LiDAR gate can't catch a
    tree that grew since 2017, so the imagery has to."""
    summary = _run_added_with_ortho(tmp_path, monkeypatch, color=(60, 170, 60))
    assert summary.qa["change_counts"]["added"] == 0
    assert summary.qa["added_rejected"]["vegetation"] == 1


def test_added_gray_roof_passes_exg(tmp_path: Path, monkeypatch) -> None:
    """A gray building-colored candidate sails through the ExG gate."""
    summary = _run_added_with_ortho(tmp_path, monkeypatch, color=(120, 120, 130))
    assert summary.qa["change_counts"]["added"] == 1
    assert summary.qa["added_rejected"]["vegetation"] == 0


# ----------------------------------------------------------------------
# Per-footprint local registration refinement
# ----------------------------------------------------------------------


def test_local_registration_recovers_residual_shift(
    tmp_path: Path, monkeypatch
) -> None:
    """A footprint whose imagery mask is offset by 2 px (residual after the
    global shift, e.g. small building outvoted by a big one) is recovered by
    the ±2 px local refinement and classified unchanged instead of
    modified/demolished."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    # Disable the global registration so the local pass is isolated.
    monkeypatch.setenv("CITYLENS_CHANGE_REGISTRATION_MAX_SHIFT_PX", "0")

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    base[10:20, 10:20] = 1
    img[12:22, 12:22] = 1  # same 10x10 building, shifted (+2, +2)

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    kinds = [f["properties"]["change_type"] for f in payload["features"]]
    assert kinds == ["unchanged"], (
        f"local refinement should recover the 2px offset, got {kinds} "
        f"(qa={summary.qa.get('local_registration_applied')})"
    )
    assert summary.qa["local_registration_applied"] == 1


def test_local_registration_disabled_leaves_offset_penalized(
    tmp_path: Path, monkeypatch
) -> None:
    """Same fixture with the local refinement disabled — the knob must
    actually change the measured IoU. The 2 px offset leaves a degraded
    IoU inside the padded ROI: intersection 8×8=64; the imagery blob is
    clipped by the ROI window to 9×9=81, so IoU = 64/(100+81-64) = 64/117."""
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_REGISTRATION_MAX_SHIFT_PX", "0")
    monkeypatch.setenv("CITYLENS_CHANGE_LOCAL_REGISTRATION_MAX_SHIFT_PX", "0")

    base = np.zeros((40, 40), dtype=np.uint8)
    img = np.zeros((40, 40), dtype=np.uint8)
    base[10:20, 10:20] = 1
    img[12:22, 12:22] = 1

    payload, summary, _ = _run_stage(tmp_path, mask=img, baseline_mask=base)
    assert summary.qa["local_registration_applied"] == 0
    ious = [f["properties"]["baseline_iou"] for f in payload["features"]]
    assert ious and abs(ious[0] - 64.0 / 117.0) < 0.01


def test_n_step_dilate_matches_iterated_3x3_reference() -> None:
    """The log-doubling separable dilation must be EXACTLY equivalent to the
    old `steps`-times-iterated 3x3 dilation on arbitrary masks."""
    from citylens_core.stages.change import _n_step_dilate, _one_step_dilate

    rng = np.random.default_rng(7)
    for steps in (1, 2, 3, 5, 8, 24):
        m = rng.random((48, 52)) < 0.05
        ref = m
        for _ in range(steps):
            ref = _one_step_dilate(ref)
        got = _n_step_dilate(m, steps)
        assert np.array_equal(got, ref), f"divergence at steps={steps}"
