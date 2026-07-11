from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio import open as rio_open
from rasterio.crs import CRS
from PIL import Image


def _write_ortho(tmp_path: Path, shape=(64, 64)) -> Path:
    path = tmp_path / "orthophoto.png"
    Image.new("RGB", shape[::-1], color=(128, 128, 128)).save(path)
    return path


def _write_geo_ortho_tif(tmp_path: Path, shape=(64, 64)) -> Path:
    path = tmp_path / "orthophoto.tif"
    arr = np.full((3, *shape), 128, dtype=np.uint8)
    with rio_open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(3857),
        transform=Affine.identity(),
    ) as dst:
        dst.write(arr)
    return path


def _write_baseline_footprints_geojson(
    tmp_path: Path, *, features: list[dict]
) -> Path:
    path = tmp_path / "baseline_footprints.geojson"
    # Include a pixel-space CRS hint so _load_baseline_footprints_mask
    # rasterizes even when the test has no ortho transform.
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "crs": "pixel", "features": features}
        )
    )
    return path


def _write_current_footprints_geojson(
    tmp_path: Path, *, features: list[dict]
) -> Path:
    path = tmp_path / "current_footprints.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "EPSG:3857"},
                },
                "features": features,
            }
        )
    )
    return path


def test_mode_prompted_without_baseline_raises(tmp_path: Path, monkeypatch) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.segment import stage_segment

    _write_ortho(tmp_path)
    monkeypatch.setenv("CITYLENS_SAM2_MODE", "prompted")
    # No baseline_footprints.geojson in work_dir.

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    with pytest.raises(RuntimeError, match="baseline_footprints.geojson"):
        stage_segment(req, tmp_path, {}, summary)


def test_mode_auto_fallback_calls_auto_when_no_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)

    calls = {"auto": 0, "prompted": 0}

    def fake_auto(image_rgb, *, cfg_path, ckpt_path, device=None):
        calls["auto"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fake_auto)
    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.delenv("CITYLENS_SAM2_MODE", raising=False)

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["previews"]
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls["auto"] == 1
    assert calls["prompted"] == 0
    assert summary.qa.get("sam2_mode") == "auto"
    assert "mask" in out


def test_mode_auto_fallback_uses_prompted_when_baseline_exists(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    # A single square footprint in pixel space covering 30x30 of the 64x64 image.
    _write_baseline_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[16, 16], [46, 16], [46, 46], [16, 46], [16, 16]]],
                },
            }
        ],
    )

    calls = {"auto": 0, "prompted": 0}
    captured_baseline = {}

    def fake_auto(image_rgb, *, cfg_path, ckpt_path, device=None):
        calls["auto"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        captured_baseline["mask"] = np.asarray(baseline_mask)
        # Return the baseline-as-mask so change-path smoke tests stay trivial.
        return np.asarray(baseline_mask, dtype=np.uint8)

    def fake_prompted_with_discovery(
        image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None
    ):
        prompted = fake_prompted(
            image_rgb,
            baseline_mask,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            device=device,
        )
        discovery = fake_auto(
            image_rgb,
            cfg_path=cfg_path,
            ckpt_path=ckpt_path,
            device=device,
        )
        return prompted, discovery

    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fake_auto)
    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setattr(
        seg_mod,
        "run_sam2_prompted_with_discovery",
        fake_prompted_with_discovery,
    )
    monkeypatch.delenv("CITYLENS_SAM2_MODE", raising=False)

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls["prompted"] == 1
    # One current-image AMG call supplies the separate added-building
    # discovery mask.  There is no baseline-image AMG call: footprints are
    # the authoritative baseline ground truth.
    assert calls["auto"] == 1
    assert summary.qa.get("sam2_mode") == "prompted"
    assert summary.qa.get("sam2_added_discovery_mode") == "automatic"
    assert summary.qa.get("sam2_added_discovery_status") == "ok"

    # The baseline we passed to the prompted predictor matches the one we
    # emit as the change-detection reference.
    assert captured_baseline["mask"].any()
    assert out["baseline_mask"] is not None
    assert np.array_equal(out["baseline_mask"], captured_baseline["mask"])
    # The prompted classification mask and automatic discovery mask remain
    # separate all the way out of segmentation.
    assert np.array_equal(out["mask"], captured_baseline["mask"])
    assert not out["added_discovery_mask"].any()
    assert out["added_discovery_mask_path"] == tmp_path / "mask_added_discovery.png"
    assert out["added_discovery_mask_path"].exists()
    assert summary.qa["sam2_added_discovery_raw"] == {
        "pixels": 0,
        "coverage_fraction": 0.0,
        "component_count": 0,
        "largest_component_pixels": 0,
        "largest_component_fraction": 0.0,
    }


def test_prompted_change_prefers_staged_current_footprints_over_amg(
    tmp_path: Path, monkeypatch
) -> None:
    from affine import Affine

    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    footprint = {
        "type": "Feature",
        "properties": {"construction_year": 2022},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[16, 16], [46, 16], [46, 46], [16, 46], [16, 16]]
            ],
        },
    }
    _write_baseline_footprints_geojson(tmp_path, features=[footprint])
    _write_current_footprints_geojson(tmp_path, features=[footprint])
    calls = {"prompted": 0, "paired": 0, "auto": 0}

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        return np.asarray(baseline_mask, dtype=np.uint8)

    def fail_paired(*args, **kwargs):
        calls["paired"] += 1
        raise AssertionError("semantic current footprints must suppress paired AMG")

    def fail_auto(*args, **kwargs):
        calls["auto"] += 1
        raise AssertionError("semantic current footprints must suppress AMG")

    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setattr(seg_mod, "run_sam2_prompted_with_discovery", fail_paired)
    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fail_auto)
    monkeypatch.delenv("CITYLENS_SAM2_MODE", raising=False)

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    out = seg_mod.stage_segment(
        req,
        tmp_path,
        {"orthophoto_transform": Affine.identity()},
        summary,
    )

    assert calls == {"prompted": 1, "paired": 0, "auto": 0}
    assert out["added_discovery_mask"] is None
    assert summary.qa["current_footprints_staged"] is True
    assert summary.qa["current_footprints_semantic_available"] is True
    assert summary.qa["sam2_added_discovery_mode"] == "current_footprints"
    assert summary.qa["sam2_added_discovery_status"] == "not_needed"


def test_prompted_added_discovery_can_be_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    _write_baseline_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[16, 16], [46, 16], [46, 46], [16, 46], [16, 16]]
                    ],
                },
            }
        ],
    )

    calls = {"auto": 0, "prompted": 0}

    def fake_auto(image_rgb, *, cfg_path, ckpt_path, device=None):
        calls["auto"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        return np.asarray(baseline_mask, dtype=np.uint8)

    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fake_auto)
    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setenv("CITYLENS_SAM2_MODE", "prompted")
    monkeypatch.setenv("CITYLENS_SAM2_ADDED_DISCOVERY", "false")

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls == {"auto": 0, "prompted": 1}
    assert out["added_discovery_mask"] is None
    assert out["added_discovery_mask_path"] is None
    assert summary.qa["sam2_added_discovery_mode"] == "disabled"
    assert summary.qa["sam2_added_discovery_status"] == "disabled"


def test_prompted_change_fails_honestly_when_added_discovery_fails(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.sam.sam2_runner import Sam2UnavailableError
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    _write_baseline_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[16, 16], [46, 16], [46, 46], [16, 46], [16, 16]]
                    ],
                },
            }
        ],
    )

    def fail_paired(*args, **kwargs):
        raise Sam2UnavailableError("automatic generator failed")

    monkeypatch.setattr(seg_mod, "run_sam2_prompted_with_discovery", fail_paired)
    monkeypatch.setenv("CITYLENS_SAM2_MODE", "prompted")

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    with pytest.raises(RuntimeError, match="SAM2 unavailable"):
        seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert summary.qa["sam2_added_discovery_mode"] == "automatic"
    assert summary.qa["sam2_added_discovery_status"] == "failed"


def test_prompted_preview_only_does_not_run_added_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    _write_baseline_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[16, 16], [46, 16], [46, 46], [16, 46], [16, 16]]
                    ],
                },
            }
        ],
    )
    calls = {"prompted": 0, "paired": 0}

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        return np.asarray(baseline_mask, dtype=np.uint8)

    def fail_paired(*args, **kwargs):
        calls["paired"] += 1
        raise AssertionError("preview-only run must not invoke automatic discovery")

    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setattr(seg_mod, "run_sam2_prompted_with_discovery", fail_paired)
    monkeypatch.setenv("CITYLENS_SAM2_MODE", "prompted")

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["previews"]
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls == {"prompted": 1, "paired": 0}
    assert out["added_discovery_mask"] is None
    assert summary.qa["sam2_added_discovery_mode"] == "not_requested"
    assert summary.qa["sam2_added_discovery_status"] == "not_requested"


def test_mode_auto_always_uses_amg_even_with_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    import citylens_core.stages.segment as seg_mod

    _write_ortho(tmp_path)
    _write_baseline_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]],
                },
            }
        ],
    )

    calls = {"auto": 0, "prompted": 0}

    def fake_auto(image_rgb, *, cfg_path, ckpt_path, device=None):
        calls["auto"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def fake_prompted(image_rgb, baseline_mask, *, cfg_path, ckpt_path, device=None):
        calls["prompted"] += 1
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fake_auto)
    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setenv("CITYLENS_SAM2_MODE", "auto")

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls["auto"] == 1
    assert calls["prompted"] == 0
    assert summary.qa.get("sam2_mode") == "auto"
    assert summary.qa.get("sam2_added_discovery_mode") == "primary_auto"
    assert summary.qa.get("sam2_added_discovery_status") == "reused"
    assert out["added_discovery_mask"] is out["mask"]
    assert out["added_discovery_mask_path"] == out["mask_path"]


def test_prompted_with_discovery_builds_model_once(monkeypatch) -> None:
    """The paired path shares one checkpoint/model load across both masks."""
    import citylens_core.sam.sam2_runner as runner

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    baseline = np.zeros((16, 16), dtype=np.uint8)
    baseline[2:8, 2:8] = 1
    calls = {"build": 0, "prompted": 0, "auto": 0, "empty_cache": 0}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls["empty_cache"] += 1

    class FakeTorch:
        cuda = FakeCuda()

    model = object()

    def fake_build(cfg_path, ckpt_path, device=None):
        calls["build"] += 1
        return FakeTorch(), model

    def fake_prompted(
        image_rgb,
        baseline_mask,
        *,
        cfg_path,
        ckpt_path,
        device=None,
        _runtime=None,
    ):
        calls["prompted"] += 1
        assert _runtime is not None
        assert isinstance(_runtime[0], FakeTorch)
        assert _runtime[1] is model
        return np.asarray(baseline_mask, dtype=np.uint8)

    def fake_auto(
        image_rgb,
        *,
        cfg_path,
        ckpt_path,
        device=None,
        _runtime=None,
    ):
        calls["auto"] += 1
        assert _runtime is not None
        assert isinstance(_runtime[0], FakeTorch)
        assert _runtime[1] is model
        return np.ones(image_rgb.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(runner, "_build_sam2_model", fake_build)
    monkeypatch.setattr(runner, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.setattr(runner, "run_sam2_auto_mask", fake_auto)

    prompted, discovery = runner.run_sam2_prompted_with_discovery(
        image,
        baseline,
        cfg_path=Path("cfg.yaml"),
        ckpt_path=Path("weights.pt"),
    )

    assert np.array_equal(prompted, baseline)
    assert discovery.all()
    assert calls == {"build": 1, "prompted": 1, "auto": 1, "empty_cache": 1}


def test_refine_cleans_added_discovery_independently(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.refine import stage_refine

    primary = np.zeros((24, 24), dtype=np.uint8)
    primary[2:9, 2:9] = 1
    discovery = np.zeros((24, 24), dtype=np.uint8)
    discovery[14:21, 14:21] = 1
    discovery[11, 11] = 1  # cleanup should remove this one-pixel AMG noise

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    out = stage_refine(
        req,
        tmp_path,
        {"mask": primary, "added_discovery_mask": discovery},
        summary,
    )

    assert out["refined_mask"][2:9, 2:9].all()
    assert not out["refined_mask"][14:21, 14:21].any()
    assert out["refined_added_discovery_mask"][14:21, 14:21].all()
    assert not out["refined_added_discovery_mask"][11, 11]
    assert summary.qa["sam2_added_discovery_refined"]["component_count"] == 1


def test_refine_rasterizes_staged_current_footprints(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.refine import stage_refine

    ortho_path = _write_geo_ortho_tif(tmp_path)
    _write_current_footprints_geojson(
        tmp_path,
        features=[
            {
                "type": "Feature",
                "properties": {"construction_year": 2022},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]
                    ],
                },
            }
        ],
    )
    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    out = stage_refine(
        req,
        tmp_path,
        {
            "mask": np.zeros((64, 64), dtype=np.uint8),
            "orthophoto_path": ortho_path,
            "orthophoto_transform": Affine.identity(),
            "orthophoto_crs": "EPSG:3857",
        },
        summary,
    )

    assert not out["refined_mask"].any()
    assert out["current_footprints_path"] == tmp_path / "current_footprints.geojson"
    assert int(out["current_footprints_mask"].sum()) == 100
    assert summary.qa["current_footprints_mask"]["component_count"] == 1


def test_run_sam2_baseline_prompted_shape_check() -> None:
    """Shape-mismatch is a programmer error and should fail loudly."""
    from citylens_core.sam.sam2_runner import run_sam2_baseline_prompted

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    bad_mask = np.ones((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        run_sam2_baseline_prompted(
            img,
            bad_mask,
            cfg_path=Path("nonexistent.yaml"),
            ckpt_path=Path("nonexistent.pt"),
        )


def test_run_sam2_baseline_prompted_empty_baseline() -> None:
    """Empty baseline -> empty mask, no SAM2 call needed."""
    from citylens_core.sam.sam2_runner import run_sam2_baseline_prompted

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    empty = np.zeros((64, 64), dtype=bool)
    out = run_sam2_baseline_prompted(
        img,
        empty,
        cfg_path=Path("nonexistent.yaml"),
        ckpt_path=Path("nonexistent.pt"),
    )
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8
    assert not out.any()
