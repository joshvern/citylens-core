from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio import open as rio_open
from rasterio.crs import CRS
from rasterio.transform import Affine


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (4, 4), color=color).save(path)


def test_stage_fetch_downloads_url_inputs_into_work_dir(tmp_path: Path, monkeypatch) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.fetch import stage_fetch
    import citylens_core.stages.fetch as fetch_mod

    ortho_bytes = tmp_path / "ortho-source.png"
    base_bytes = tmp_path / "base-source.png"
    _write_png(ortho_bytes, (120, 120, 120))
    _write_png(base_bytes, (100, 100, 100))

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.headers = {"content-type": "image/png"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            yield self.payload

    payloads = {
        "https://example.test/ortho.png": ortho_bytes.read_bytes(),
        "https://example.test/base.png": base_bytes.read_bytes(),
    }

    def fake_get(url: str, stream: bool = True, timeout: int = 120):
        return FakeResponse(payloads[url])

    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    req = CitylensRequest(
        address="x",
        segmentation_backend="sam2",
        orthophoto_url="https://example.test/ortho.png",
        baseline_url="https://example.test/base.png",
        outputs=["change"],
    )
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    # Drop a fake lidar.las into the work_dir to exercise the lidar-hash path.
    (tmp_path / "lidar.las").write_bytes(b"lidar-bytes")

    ctx = stage_fetch(req, tmp_path, {}, summary)

    assert ctx["orthophoto_path"] == tmp_path / "orthophoto.png"
    assert ctx["baseline_path"] == tmp_path / "baseline.png"
    assert (tmp_path / "orthophoto.png").read_bytes() == payloads["https://example.test/ortho.png"]
    assert (tmp_path / "baseline.png").read_bytes() == payloads["https://example.test/base.png"]

    import hashlib

    assert ctx["orthophoto_sha256"] == hashlib.sha256(
        payloads["https://example.test/ortho.png"]
    ).hexdigest()
    assert ctx["baseline_sha256"] == hashlib.sha256(
        payloads["https://example.test/base.png"]
    ).hexdigest()
    assert ctx["lidar_sha256"] == hashlib.sha256(b"lidar-bytes").hexdigest()


def test_change_stage_emits_georeferenced_geojson_when_metadata_exists(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    # Per-building classification now gates features by min-area; the
    # georeferenced path uses m² not px, so override both.
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_M2", "1")
    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    # Isolated 3x3 current-year building with an empty baseline — should be
    # classified as `added`. Transform puts pixel (0,0) at world (100,200)
    # with 2m/px scale, so the feature spans world x [100..106], y [200..194].
    imagery = np.zeros((5, 5), dtype=np.uint8)
    imagery[0:3, 0:3] = 1
    ctx = {
        "mask": imagery,
        "baseline_mask": np.zeros((5, 5), dtype=np.uint8),
        "orthophoto_transform": Affine.translation(100, 200) * Affine.scale(2, -2),
        "orthophoto_crs": CRS.from_epsg(3857).to_string(),
    }

    out = stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())

    assert out["change_path"] == tmp_path / "change.geojson"
    feats = payload["features"]
    assert len(feats) == 1
    f = feats[0]
    assert f["properties"]["change_type"] == "added"
    assert f["properties"]["crs"] == "EPSG:3857"
    # `added` features don't carry a baseline_iou (no prior building to compare to).
    assert f["properties"].get("baseline_iou") is None
    assert f["geometry"]["type"] == "Polygon"

    ring = f["geometry"]["coordinates"][0]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    assert min(xs) == 100.0 and max(xs) == 106.0
    assert min(ys) == 194.0 and max(ys) == 200.0


def test_refine_stage_uses_baseline_footprints_guidance(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.refine import stage_refine

    ortho_path = tmp_path / "orthophoto.tif"
    data = np.zeros((4, 4), dtype=np.uint8)
    transform = Affine.translation(100, 200) * Affine.scale(2, -2)
    with rio_open(
        ortho_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(3857),
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    baseline_footprints = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"kind": "building"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[100, 200], [106, 200], [106, 194], [100, 194], [100, 200]]],
                },
            }
        ],
    }
    (tmp_path / "baseline_footprints.geojson").write_text(json.dumps(baseline_footprints))

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))
    mask = np.pad(np.ones((3, 3), dtype=np.uint8), ((1, 6), (1, 6)))
    mask[8, 8] = 1
    ctx = {
        "mask": mask,
        "baseline_mask": np.zeros((10, 10), dtype=np.uint8),
        "orthophoto_path": ortho_path,
        "orthophoto_transform": transform,
        "orthophoto_crs": CRS.from_epsg(3857).to_string(),
    }

    out = stage_refine(req, tmp_path, ctx, summary)

    assert out["baseline_footprints_path"] == tmp_path / "baseline_footprints.geojson"
    assert out["baseline_footprints_mask"].shape == (10, 10)
    assert out["baseline_footprints_mask"].sum() == 4
    assert out["refined_baseline_mask"].sum() == 4
    assert out["refined_mask"].sum() == 9


def test_reconstruct_uses_lidar_when_available(tmp_path: Path, monkeypatch) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.reconstruct import stage_reconstruct

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc))

    (tmp_path / "lidar.las").write_bytes(b"fake")

    class FakeLas:
        x = np.array([0.1, 1.1, 0.1, 1.1], dtype=np.float32)
        y = np.array([0.1, 0.1, 1.1, 1.1], dtype=np.float32)
        z = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    fake_module = SimpleNamespace(read=lambda path: FakeLas())
    monkeypatch.setitem(sys.modules, "laspy", fake_module)

    ctx = {
        "mask": np.ones((2, 2), dtype=np.uint8),
        "orthophoto_transform": Affine.identity(),
    }

    out = stage_reconstruct(req, tmp_path, ctx, summary)
    mesh_text = (tmp_path / "mesh.ply").read_text()

    assert out["mesh_path"] == tmp_path / "mesh.ply"
    assert out["mesh_height_source"] == "lidar"
    assert out["mesh_footprint_mask"].shape == (2, 2)
    assert "10.0" in mesh_text
    assert "20.0" in mesh_text
    assert "30.0" in mesh_text
    assert "40.0" in mesh_text
