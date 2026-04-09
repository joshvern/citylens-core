from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_dummy_rgb(path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(path)


def test_missing_inputs_fails_and_writes_only_summary(tmp_path: Path, monkeypatch) -> None:
    from citylens_core import CitylensRequest, run_citylens
    import citylens_core.sam.assets as sam_assets

    monkeypatch.setattr(sam_assets, "ensure_sam2_assets", lambda *args, **kwargs: None)

    req = CitylensRequest(address="test", segmentation_backend="sam2", outputs=["previews", "change", "mesh"])
    artifacts = run_citylens(req, tmp_path)

    assert set(artifacts.keys()) == {"summary"}
    assert artifacts["summary"].exists()
    assert not (tmp_path / "preview.png").exists()
    assert not (tmp_path / "change.geojson").exists()
    assert not (tmp_path / "mesh.ply").exists()

    payload = json.loads((tmp_path / "run_summary.json").read_text())
    assert payload.get("ok") is False
    assert payload.get("error_code") == "missing_dependency"
    assert isinstance(payload.get("missing_paths"), list)


def test_missing_sam2_weights_fails_no_model_artifacts(tmp_path: Path) -> None:
    """Required by spec: missing weights => failure and no output artifacts written."""

    from citylens_core import CitylensRequest, run_citylens

    # Provide real input images so the failure is specifically about SAM2 assets.
    ortho = tmp_path / "ortho.png"
    base = tmp_path / "base.png"
    _write_dummy_rgb(ortho)
    _write_dummy_rgb(base)

    req = CitylensRequest(
        address="test",
        segmentation_backend="sam2",
        orthophoto_path=ortho,
        baseline_path=base,
        sam2_cfg="configs/sam2.1/nonexistent.yaml",
        sam2_checkpoint="weights/nonexistent.pt",
        outputs=["previews", "change", "mesh"],
    )

    artifacts = run_citylens(req, tmp_path)

    assert set(artifacts.keys()) == {"summary"}
    assert (tmp_path / "run_summary.json").exists()
    assert not (tmp_path / "preview.png").exists()
    assert not (tmp_path / "change.geojson").exists()
    assert not (tmp_path / "mesh.ply").exists()

    payload = json.loads((tmp_path / "run_summary.json").read_text())
    assert payload.get("ok") is False
    assert payload.get("error_code") == "missing_dependency"
    assert payload.get("error_message")
    assert "SAM2" in payload.get("error_message", "")
