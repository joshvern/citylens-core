from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .io.artifacts import compute_artifact_metadata, write_run_summary
from .models import CitylensRequest, PipelineSummary
from .stages.change import stage_change
from .stages.fetch import stage_fetch
from .stages.reconstruct import stage_reconstruct
from .stages.render import stage_render
from .stages.resolve import stage_resolve
from .stages.segment import stage_segment

logger = logging.getLogger("citylens_core.pipeline")

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

    Produces real artifacts on success.

    On failure, writes only `run_summary.json` with structured error information
    (error_code, error_message, missing_paths) and returns a map containing just
    that summary artifact.
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

    def _log(event: str, payload: dict[str, Any]) -> None:
        try:
            logger.info(json.dumps({"event": event, **payload}, sort_keys=True))
        except Exception:
            # Never let logging break pipeline execution.
            logger.info("%s %s", event, payload)

    def _fail(*, code: str, message: str, missing_paths: Optional[list[str]] = None) -> dict[str, Path]:
        summary.fail(error_code=code, error_message=message, missing_paths=missing_paths)
        summary.finished_at = datetime.utcnow()
        duration_s = max(0.0, float(time.time() - t0))
        write_run_summary(summary, standard_paths["summary"], extra={"duration_s": duration_s})
        _emit_progress(progress_cb, 100, "done")
        return {"summary": standard_paths["summary"]}

    # Preflight required inputs/assets before any expensive stage work.
    try:
        # Inputs: require both orthophoto and baseline if change output is requested.
        missing_inputs: list[str] = []
        ortho_in = req.orthophoto_path or (work_dir / "orthophoto.png")
        base_in = req.baseline_path or (work_dir / "baseline.png")
        if not Path(ortho_in).exists():
            missing_inputs.append(str(Path(ortho_in)))
        if want_change and not Path(base_in).exists():
            missing_inputs.append(str(Path(base_in)))
        if missing_inputs:
            return _fail(
                code="missing_dependency",
                message="Missing required input data",
                missing_paths=missing_inputs,
            )

        # Weights/configs for SAM2
        if req.segmentation_backend == "sam2":
            from .sam.assets import Sam2AssetsMissingError, ensure_sam2_assets

            try:
                ensure_sam2_assets(Path(req.sam2_cfg or ""), Path(req.sam2_checkpoint or ""))
            except Sam2AssetsMissingError as e:
                msg = str(e)
                # Best-effort parse: ensure_sam2_assets includes absolute paths in message.
                missing: list[str] = []
                if ":" in msg:
                    after = msg.split(":", 1)[1]
                    for part in after.split(","):
                        p = part.strip().split(".")[0].strip()
                        if p.startswith("/"):
                            missing.append(p)
                return _fail(
                    code="missing_dependency",
                    message="SAM2 assets missing",
                    missing_paths=missing or None,
                )
    except Exception as e:
        return _fail(code="pipeline_error", message=f"preflight failed: {type(e).__name__}: {e}")

    def _run_stage(name: str, pct: int, fn, ctx: dict[str, Any]) -> dict[str, Any]:
        _emit_progress(progress_cb, pct, name)
        _log("stage_started", {"stage": name})
        t_stage0 = time.perf_counter()
        try:
            out = fn(req, work_dir, ctx, summary)
            summary.stage_status[name] = "ok"
            _log(
                "stage_finished",
                {"stage": name, "duration_secs": float(time.perf_counter() - t_stage0), "ok": True},
            )
            return out
        except Exception as e:
            summary.stage_status[name] = "error"
            _log(
                "stage_finished",
                {
                    "stage": name,
                    "duration_secs": float(time.perf_counter() - t_stage0),
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise

    def _skip_stage(name: str, pct: int, ctx: dict[str, Any]) -> dict[str, Any]:
        _emit_progress(progress_cb, pct, name)
        summary.stage_status[name] = "skipped"
        _log("stage_skipped", {"stage": name, "ok": True})
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
    except FileNotFoundError as e:
        return _fail(code="missing_dependency", message=str(e), missing_paths=None)
    except Exception as e:
        return _fail(code="pipeline_error", message=f"{type(e).__name__}: {e}", missing_paths=None)

    # Success path: ensure required artifacts exist and record metadata.
    required = ["preview", "change", "mesh"]
    missing_required: list[str] = []
    for key in required:
        p = standard_paths[key]
        if not p.exists():
            missing_required.append(str(p))
    if missing_required:
        return _fail(
            code="pipeline_error",
            message="Pipeline succeeded but required artifacts were not written",
            missing_paths=missing_required,
        )

    # Write run_summary last, then include artifact metadata for all four.
    summary.finished_at = datetime.utcnow()
    duration_s = max(0.0, float(time.time() - t0))
    write_run_summary(summary, standard_paths["summary"], extra={"duration_s": duration_s})

    for key, path in standard_paths.items():
        if path.exists():
            try:
                summary.artifacts[key] = compute_artifact_metadata(key, path)
            except Exception as e:
                summary.warn(f"artifact-metadata:{key}: {type(e).__name__}: {e}")
    # Re-write summary with artifact metadata included.
    write_run_summary(summary, standard_paths["summary"], extra={"duration_s": duration_s})

    _emit_progress(progress_cb, 100, "done")
    return standard_paths
