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


def test_outputs_gating_previews_only_does_not_require_change_or_mesh(tmp_path: Path, monkeypatch) -> None:
    import citylens_core.pipeline as pl

    # Avoid heavy deps: monkeypatch stages to only emit preview.
    monkeypatch.setattr(pl, "stage_resolve", lambda req, wd, ctx, summary: ctx)
    monkeypatch.setattr(pl, "stage_fetch", lambda req, wd, ctx, summary: {**ctx, "orthophoto_path": Path(wd) / "orthophoto.png"})
    monkeypatch.setattr(pl, "stage_segment", lambda req, wd, ctx, summary: ctx)

    def _stage_render(req, wd, ctx, summary):
        p = Path(wd) / "preview.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return ctx

    monkeypatch.setattr(pl, "stage_render", _stage_render)

    from citylens_core.models import CitylensRequest

    work_dir = tmp_path
    (work_dir / "orthophoto.png").write_bytes(b"x")
    req = CitylensRequest(address="x", segmentation_backend="unet", outputs=["previews"])
    out = pl.run_citylens(req, work_dir)

    assert set(out.keys()) == {"preview", "summary"}
    assert (work_dir / "preview.png").exists()
    assert (work_dir / "run_summary.json").exists()
    assert not (work_dir / "change.geojson").exists()
    assert not (work_dir / "mesh.ply").exists()
