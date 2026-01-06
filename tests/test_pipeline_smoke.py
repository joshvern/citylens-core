from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_unimplemented_backend_fails_without_placeholders(tmp_path: Path) -> None:
    from citylens_core import CitylensRequest, run_citylens

    # Provide inputs so failure is specifically due to backend support.
    Image = pytest.importorskip("PIL.Image")
    ortho = tmp_path / "ortho.png"
    base = tmp_path / "base.png"
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(ortho)
    Image.new("RGB", (32, 32), color=(110, 110, 110)).save(base)

    req = CitylensRequest(
        address="test",
        segmentation_backend="unet",
        orthophoto_path=ortho,
        baseline_path=base,
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
