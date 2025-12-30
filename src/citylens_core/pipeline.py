from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .io.artifacts import compute_artifact_metadata, ensure_standard_artifacts, write_run_summary
from .models import CitylensRequest, PipelineSummary
from .stages.change import stage_change
from .stages.fetch import stage_fetch
from .stages.reconstruct import stage_reconstruct
from .stages.render import stage_render
from .stages.resolve import stage_resolve
from .stages.segment import stage_segment

ProgressCb = Optional[Callable[..., Any]]


def _emit_progress(progress_cb: ProgressCb, pct: int, stage: str) -> None:
    if not callable(progress_cb):
        return
    try:
        progress_cb(pct, stage)
    except TypeError:
        progress_cb(pct)


def run_citylens(request: Any, work_dir: Path, progress_cb: ProgressCb = None) -> dict[str, Path]:
    """Orchestrate the Citylens pipeline.

    Progress points: resolve 5, fetch 25, segment 55, change 75, reconstruct 90,
    render 95, done 100.

    Always ensures the standard artifacts exist in `work_dir`, even on failures.
    """

    work_dir = Path(work_dir)
    req = request if isinstance(request, CitylensRequest) else CitylensRequest.model_validate(request)
    summary = PipelineSummary(request=req, work_dir=work_dir, started_at=datetime.utcnow())
    t0 = time.time()

    standard_paths = {
        "preview": work_dir / "preview.png",
        "change": work_dir / "change.geojson",
        "mesh": work_dir / "mesh.ply",
        "summary": work_dir / "run_summary.json",
    }

    outputs = {str(o).strip().lower() for o in (req.outputs or []) if str(o).strip()}
    want_preview = ("previews" in outputs) or ("preview" in outputs)
    want_change = "change" in outputs
    want_mesh = "mesh" in outputs

    def _run_stage(name: str, pct: int, fn, ctx: dict[str, Any]) -> dict[str, Any]:
        _emit_progress(progress_cb, pct, name)
        try:
            out = fn(req, work_dir, ctx, summary)
            summary.stage_status[name] = "ok"
            return out
        except Exception as e:  # best-effort pipeline
            summary.stage_status[name] = "error"
            summary.error(f"stage:{name}: {type(e).__name__}: {e}")
            return ctx

    def _skip_stage(name: str, pct: int, ctx: dict[str, Any]) -> dict[str, Any]:
        _emit_progress(progress_cb, pct, name)
        summary.stage_status[name] = "skipped"
        return ctx

    ctx: dict[str, Any] = {}
    try:
        ctx = _run_stage("resolve", 5, stage_resolve, ctx)
        ctx = _run_stage("fetch", 25, stage_fetch, ctx)
        ctx = _run_stage("segment", 55, stage_segment, ctx)

        if want_change:
            ctx = _run_stage("change", 75, stage_change, ctx)
        else:
            ctx = _skip_stage("change", 75, ctx)

        if want_mesh:
            ctx = _run_stage("reconstruct", 90, stage_reconstruct, ctx)
        else:
            ctx = _skip_stage("reconstruct", 90, ctx)

        if want_preview:
            ctx = _run_stage("render", 95, stage_render, ctx)
        else:
            ctx = _skip_stage("render", 95, ctx)
    finally:
        ensure_standard_artifacts(work_dir)

        for key, path in standard_paths.items():
            if path.exists():
                try:
                    summary.artifacts[key] = compute_artifact_metadata(key, path)
                except Exception as e:
                    summary.warn(f"artifact-metadata:{key}: {type(e).__name__}: {e}")

        summary.finished_at = datetime.utcnow()
        duration_s = max(0.0, float(time.time() - t0))
        write_run_summary(summary, standard_paths["summary"], extra={"duration_s": duration_s})

        _emit_progress(progress_cb, 100, "done")

    return standard_paths
