from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("backend", ["unet", "smp"])
def test_unsupported_backends_are_rejected_at_validation(backend: str) -> None:
    from citylens_core.models import CitylensRequest

    with pytest.raises(ValidationError):
        CitylensRequest.model_validate({"address": "test", "segmentation_backend": backend})


def test_outputs_gating_previews_only_does_not_require_change_or_mesh(tmp_path: Path, monkeypatch) -> None:
    import citylens_core.pipeline as pl
    import citylens_core.sam.assets as sam_assets

    monkeypatch.setattr(sam_assets, "ensure_sam2_assets", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl, "stage_resolve", lambda req, wd, ctx, summary: {**ctx, "work_dir": wd})

    def _stage_fetch(req, wd, ctx, summary):
        return {**ctx, "orthophoto_path": Path(wd) / "orthophoto.png"}

    def _stage_segment(req, wd, ctx, summary):
        return {**ctx, "mask": np.ones((4, 4), dtype=np.uint8)}

    def _stage_render(req, wd, ctx, summary):
        p = Path(wd) / "preview.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return ctx

    monkeypatch.setattr(pl, "stage_fetch", _stage_fetch)
    monkeypatch.setattr(pl, "stage_segment", _stage_segment)
    monkeypatch.setattr(pl, "stage_refine", lambda req, wd, ctx, summary: {**ctx, "refined_mask": ctx["mask"], "refined_baseline_mask": None})
    monkeypatch.setattr(pl, "stage_change", lambda req, wd, ctx, summary: ctx)
    monkeypatch.setattr(pl, "stage_reconstruct", lambda req, wd, ctx, summary: ctx)
    monkeypatch.setattr(pl, "stage_render", _stage_render)

    from citylens_core.models import CitylensRequest

    work_dir = tmp_path
    (work_dir / "orthophoto.png").write_bytes(b"x")
    req = CitylensRequest(address="x", segmentation_backend="sam2", outputs=["previews"])
    out = pl.run_citylens(req, work_dir)

    assert set(out.keys()) == {"preview", "summary"}
    assert (work_dir / "preview.png").exists()
    assert (work_dir / "run_summary.json").exists()
    assert not (work_dir / "change.geojson").exists()
    assert not (work_dir / "mesh.ply").exists()

    payload = json.loads((work_dir / "run_summary.json").read_text())
    assert payload.get("ok") is True
