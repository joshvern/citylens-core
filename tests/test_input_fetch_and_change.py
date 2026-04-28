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


def test_reconstruct_lod1_emits_per_building_extrusions(
    tmp_path: Path, monkeypatch
) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.reconstruct import stage_reconstruct

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    change_path = tmp_path / "change.geojson"
    change_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"change_type": "unchanged"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"change_type": "demolished"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[5, 5], [9, 5], [9, 9], [5, 9], [5, 5]]
                            ],
                        },
                    },
                ],
            }
        )
    )

    heights = np.full((10, 10), np.nan, dtype=np.float32)
    heights[0:4, 0:4] = 25.0

    ctx = {
        "mask": np.ones((10, 10), dtype=np.uint8),
        "orthophoto_transform": Affine.identity(),
        "lidar_heights": heights,
        "lidar_ground_z": 0.0,
        "change_path": change_path,
    }

    out = stage_reconstruct(req, tmp_path, ctx, summary)
    mesh_text = (tmp_path / "mesh.ply").read_text()

    assert out["mesh_path"] == tmp_path / "mesh.ply"
    assert summary.qa["mesh_source"] == "lod1"
    assert summary.qa["mesh_buildings"] == 1  # demolished skipped
    assert summary.qa["mesh_stats"]["skipped_demolished"] == 1
    # The roof samples at 25 m should round-trip through the PLY text.
    assert "25.0" in mesh_text
    # mesh_footprint_mask must be the union of polygons we actually extruded —
    # not the full grid (which would include streets/cars/trees) and not empty.
    fp = out["mesh_footprint_mask"]
    assert fp.sum() > 0
    assert fp.sum() < heights.size


def test_earclip_triangulates_concave_polygon_without_external_spokes() -> None:
    """An L-shaped polygon's centroid lands in the inner notch, OUTSIDE
    the polygon. A naive centroid-fan triangulation would emit triangles
    that stick out into the empty notch ("spokes"). Ear clipping must
    keep every triangle inside the polygon.
    """
    from citylens_core.stages.reconstruct import _earclip_triangulate

    # L-shape, CCW: 6 points outlining an inverted-L (centroid in the notch).
    #
    #   (0,0) ---- (4,0)
    #     |          |
    #     |          |
    #     |        (4,2) -------- (6,2)
    #     |                          |
    #   (0,6) -------------------- (6,6)
    ring = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 2.0),
        (6.0, 2.0),
        (6.0, 6.0),
        (0.0, 6.0),
    ]
    tris = _earclip_triangulate(ring)
    assert len(tris) == 4  # N-2 = 6-2 = 4

    # Every triangle's centroid must be inside the L-shape (no spokes).
    def _point_in_polygon(p, poly):
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside

    for ti, tj, tk in tris:
        a, b, c = ring[ti], ring[tj], ring[tk]
        cx = (a[0] + b[0] + c[0]) / 3
        cy = (a[1] + b[1] + c[1]) / 3
        assert _point_in_polygon((cx, cy), ring), (
            f"triangle ({a},{b},{c}) centroid ({cx},{cy}) lies outside the L-polygon"
        )


def test_change_features_carry_height_m_when_lidar_available(
    tmp_path: Path, monkeypatch
) -> None:
    """Each classified change feature should expose its 95th-pct LiDAR
    height-above-ground in properties.height_m so UIs can show building
    heights without the user parsing the mesh PLY.
    """
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.change import stage_change

    monkeypatch.setenv("CITYLENS_CHANGE_MIN_AREA_PX", "1")

    # Single 4x4 baseline footprint that the imagery covers exactly →
    # unchanged. LiDAR shows the building is 25 m above ground.
    base = np.zeros((10, 10), dtype=np.uint8)
    img = np.zeros((10, 10), dtype=np.uint8)
    base[2:6, 2:6] = 1
    img[2:6, 2:6] = 1

    heights = np.full((10, 10), np.nan, dtype=np.float32)
    heights[2:6, 2:6] = 30.0  # 25 m above ground at z=5

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    ctx = {
        "mask": img,
        "baseline_mask": base,
        "orthophoto_transform": Affine.identity(),
        "orthophoto_crs": "EPSG:3857",
        "lidar_heights": heights,
        "lidar_ground_z": 5.0,
    }

    out = stage_change(req, tmp_path, ctx, summary)
    payload = json.loads((tmp_path / "change.geojson").read_text())
    feats = payload["features"]
    assert len(feats) == 1
    props = feats[0]["properties"]
    assert props["change_type"] == "unchanged"
    # 25 m = 30 (roof) − 5 (ground)
    assert "height_m" in props, props
    assert abs(props["height_m"] - 25.0) < 0.5, props["height_m"]


def test_render_change_aware_preview(tmp_path: Path) -> None:
    from PIL import Image
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.stages.render import stage_render

    req = CitylensRequest(
        address="x",
        segmentation_backend="sam2",
        imagery_year=2024,
        baseline_year=2017,
    )
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )
    summary.qa["change_counts"] = {
        "unchanged": 1,
        "modified": 0,
        "demolished": 1,
        "added": 1,
    }

    # 256x256 gray orthophoto (large enough that legend+year label don't
    # overwrite the polygons we're asserting on).
    ortho_path = tmp_path / "orthophoto.png"
    Image.new("RGB", (256, 256), color=(80, 80, 80)).save(ortho_path)

    change_path = tmp_path / "change.geojson"
    change_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"change_type": "unchanged"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"change_type": "added"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[80, 80], [100, 80], [100, 100], [80, 100], [80, 80]]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"change_type": "demolished"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[130, 80], [150, 80], [150, 100], [130, 100], [130, 80]]
                            ],
                        },
                    },
                ],
            }
        )
    )

    ctx = {
        "orthophoto_path": ortho_path,
        "orthophoto_transform": Affine.identity(),
        "change_path": change_path,
        "mask": np.ones((256, 256), dtype=np.uint8),
    }

    out = stage_render(req, tmp_path, ctx, summary)
    preview = Image.open(out["preview_path"]).convert("RGBA")
    pixels = np.array(preview)

    assert summary.qa["preview_source"] == "change_classified"
    # Green-dominant pixels (added) should exist in the added region.
    added_region = pixels[82:98, 82:98]
    assert (added_region[..., 1] > added_region[..., 0]).any()
    # Red-dominant pixels (demolished) should exist in the demolished region.
    demo_region = pixels[82:98, 132:148]
    assert (demo_region[..., 0] > demo_region[..., 1]).any()


def test_change_geojson_is_reprojected_to_wgs84(tmp_path: Path) -> None:
    """Pipeline post-step rewrites change.geojson from the ortho CRS to
    WGS84 lng/lat so Leaflet renders the polygons.
    """
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.pipeline import _reproject_change_geojson_to_wgs84

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    # NYC-ish point in EPSG:3857 (Web Mercator meters); should reproject
    # to ~(-73.95, 40.65) in WGS84 lng/lat.
    change_path = tmp_path / "change.geojson"
    change_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "change_type": "added",
                            "crs": "EPSG:3857",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-8233100, 4961100],
                                    [-8233000, 4961100],
                                    [-8233000, 4961200],
                                    [-8233100, 4961200],
                                    [-8233100, 4961100],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )

    _reproject_change_geojson_to_wgs84(change_path, "EPSG:3857", summary)

    payload = json.loads(change_path.read_text())
    feat = payload["features"][0]
    coords = feat["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    # Brooklyn-area lng/lat band
    assert all(-74.1 < lon < -73.8 for lon in lons), lons
    assert all(40.5 < lat < 40.8 for lat in lats), lats
    # Original CRS is preserved on the feature for traceability
    assert feat["properties"]["crs"] == "EPSG:4326"
    assert feat["properties"]["source_crs"] == "EPSG:3857"


def test_reproject_change_geojson_noop_for_pixel_or_wgs84(tmp_path: Path) -> None:
    from citylens_core.models import CitylensRequest, PipelineSummary
    from citylens_core.pipeline import _reproject_change_geojson_to_wgs84

    req = CitylensRequest(address="x", segmentation_backend="sam2")
    summary = PipelineSummary(
        request=req, work_dir=tmp_path, started_at=datetime.now(timezone.utc)
    )

    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"change_type": "added", "crs": "pixel"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }
        ],
    }
    p = tmp_path / "change.geojson"
    p.write_text(json.dumps(payload))

    # pixel src_crs => no-op
    _reproject_change_geojson_to_wgs84(p, "pixel", summary)
    assert json.loads(p.read_text())["features"][0]["geometry"]["coordinates"][0][1] == [10, 0]

    # already-WGS84 src_crs => no-op (still won't double-translate)
    p.write_text(json.dumps(payload))
    _reproject_change_geojson_to_wgs84(p, "EPSG:4326", summary)
    assert json.loads(p.read_text())["features"][0]["geometry"]["coordinates"][0][1] == [10, 0]
