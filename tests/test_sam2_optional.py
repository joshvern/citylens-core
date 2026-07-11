import json
import importlib.util
from pathlib import Path

import pytest


def test_sam2_optional(tmp_path: Path, monkeypatch) -> None:
    sam2_spec = importlib.util.find_spec("sam2")
    if sam2_spec is None:
        pytest.skip("sam2 not installed")

    # Assets resolve against CITYLENS_ASSETS_ROOT (or cwd). Point at an empty
    # dir so this test exercises the missing-weights failure path even on dev
    # machines where `make sam2-assets` has populated the repo checkout.
    empty_assets = tmp_path / "no-assets"
    empty_assets.mkdir()
    monkeypatch.setenv("CITYLENS_ASSETS_ROOT", str(empty_assets))

    from citylens_core import CitylensRequest, run_citylens

    # Provide real input images so we can validate failure behavior without
    # producing placeholder outputs.
    try:
        from PIL import Image

        ortho = tmp_path / "ortho.png"
        base = tmp_path / "base.png"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(ortho)
        Image.new("RGB", (64, 64), color=(127, 127, 127)).save(base)
    except Exception as e:
        pytest.skip(f"PIL unavailable: {e}")

    # Intentionally point at default asset locations that may not exist.
    req = CitylensRequest(
        address="test",
        segmentation_backend="sam2",
        orthophoto_path=ortho,
        baseline_path=base,
        outputs=["previews", "change", "mesh"],
    )
    artifacts = run_citylens(req, tmp_path)

    # Missing weights should fail and only produce run_summary.json
    assert set(artifacts.keys()) == {"summary"}
    assert artifacts["summary"].exists()
    assert not (tmp_path / "preview.png").exists()
    assert not (tmp_path / "change.geojson").exists()
    assert not (tmp_path / "mesh.ply").exists()

    payload = json.loads(artifacts["summary"].read_text())
    assert payload.get("ok") is False
    assert payload.get("error_code") in ("missing_dependency", "pipeline_error")
