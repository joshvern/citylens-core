from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_run_summary_includes_qa_and_performance_fields(tmp_path: Path, monkeypatch) -> None:
    import citylens_core.pipeline as pl
    import citylens_core.sam.assets as sam_assets

    monkeypatch.setattr(sam_assets, "ensure_sam2_assets", lambda *args, **kwargs: None)

    monkeypatch.setattr(pl, "stage_resolve", lambda req, wd, ctx, summary: {**ctx, "work_dir": wd})

    def _stage_fetch(req, wd, ctx, summary):
        return {
            **ctx,
            "orthophoto_path": Path(wd) / "orthophoto.png",
            "orthophoto_transform": None,
            "orthophoto_crs": None,
            "orthophoto_sha256": "deadbeef" * 8,
            "baseline_sha256": "cafebabe" * 8,
            "lidar_sha256": "feedface" * 8,
        }

    def _stage_segment(req, wd, ctx, summary):
        return {
            **ctx,
            "mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            "baseline_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        }

    def _stage_refine(req, wd, ctx, summary):
        return {
            **ctx,
            "refined_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            "refined_baseline_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            "baseline_footprints_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            "lidar_heights": np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32),
            "lidar_ground_z": 0.0,
        }

    def _stage_reconstruct(req, wd, ctx, summary):
        mesh = Path(wd) / "mesh.ply"
        mesh.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 4",
                    "property float x",
                    "property float y",
                    "property float z",
                    "element face 2",
                    "property list uchar int vertex_indices",
                    "end_header",
                    "0 0 1",
                    "1 0 1",
                    "0 1 1",
                    "1 1 1",
                    "3 0 1 2",
                    "3 0 2 3",
                ]
            )
        )
        return {**ctx, "mesh_path": mesh, "mesh_footprint_mask": np.array([[1, 0], [0, 0]], dtype=np.uint8), "mesh_height_source": "mask"}

    def _stage_render(req, wd, ctx, summary):
        preview = Path(wd) / "preview.png"
        preview.write_bytes(b"\x89PNG\r\n\x1a\n")
        return ctx

    monkeypatch.setattr(pl, "stage_fetch", _stage_fetch)
    monkeypatch.setattr(pl, "stage_segment", _stage_segment)
    monkeypatch.setattr(pl, "stage_refine", _stage_refine)
    monkeypatch.setattr(pl, "stage_change", pl.stage_change)
    monkeypatch.setattr(pl, "stage_reconstruct", _stage_reconstruct)
    monkeypatch.setattr(pl, "stage_render", _stage_render)

    from citylens_core import CitylensRequest, run_citylens

    req = CitylensRequest(address="100 E 21st St Brooklyn, NY 11226", segmentation_backend="sam2")
    artifacts = run_citylens(req, tmp_path)

    assert set(artifacts.keys()) == {"summary", "preview", "change", "mesh"}

    payload = json.loads((tmp_path / "run_summary.json").read_text())
    assert payload["qa"]["reference_case_id"] == "100_e_21st_st_brooklyn_ny_11226"
    assert payload["qa"]["baseline_footprints_used"] is True
    assert payload["qa"]["lidar_used"] is True
    assert payload["qa"]["mask_iou"] == 1.0
    assert payload["qa"]["change_polygon_f1"] == 1.0
    assert payload["qa"]["mesh_footprint_iou"] == 1.0
    assert payload["qa"]["parity_status"] == "complete"
    assert isinstance(payload["performance"]["total_runtime_seconds"], float)
    assert isinstance(payload["performance"]["stage_timings_seconds"], dict)
    assert "resolve" in payload["performance"]["stage_timings_seconds"]

    # Input attestation: these fields prove the run consumed real bytes.
    assert payload["qa"]["sam2_used"] is True
    assert payload["qa"]["orthophoto_sha256"] == "deadbeef" * 8
    assert payload["qa"]["baseline_sha256"] == "cafebabe" * 8
    assert payload["qa"]["lidar_sha256"] == "feedface" * 8
