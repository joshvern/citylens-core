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

    ortho_path = _write_ortho(tmp_path)
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

    monkeypatch.setattr(seg_mod, "run_sam2_auto_mask", fake_auto)
    monkeypatch.setattr(seg_mod, "run_sam2_baseline_prompted", fake_prompted)
    monkeypatch.delenv("CITYLENS_SAM2_MODE", raising=False)

    req = CitylensRequest(
        address="x", segmentation_backend="sam2", outputs=["change"]
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    out = seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls["prompted"] == 1
    # Baseline path skips the 2nd SAM2 call — baseline_footprints ARE the ground truth.
    assert calls["auto"] == 0
    assert summary.qa.get("sam2_mode") == "prompted"

    # The baseline we passed to the prompted predictor matches the one we
    # emit as the change-detection reference.
    assert captured_baseline["mask"].any()
    assert out["baseline_mask"] is not None
    assert np.array_equal(out["baseline_mask"], captured_baseline["mask"])


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
        address="x", segmentation_backend="sam2", outputs=["previews"]
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    seg_mod.stage_segment(req, tmp_path, {}, summary)

    assert calls["auto"] == 1
    assert calls["prompted"] == 0
    assert summary.qa.get("sam2_mode") == "auto"


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
