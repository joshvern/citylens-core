from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .io.artifacts import compute_artifact_metadata, write_run_summary
from .io.geo import mask_f1, mask_iou
from .models import CitylensRequest, PipelineSummary
from .stages.change import stage_change
from .stages.fetch import stage_fetch
from .stages.reconstruct import stage_reconstruct
from .stages.refine import stage_refine
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


def _slugify_reference_case(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return slug.strip("_") or "unknown_case"


def _reproject_change_geojson_to_wgs84(
    change_path: Path | None,
    src_crs: str | None,
    summary: PipelineSummary,
) -> None:
    """Rewrite change.geojson in WGS84 (EPSG:4326) so it renders on
    standard web maps. Originating CRS is preserved per-feature in
    properties.source_crs.

    No-op when:
    - the file doesn't exist (change wasn't requested),
    - src_crs is missing or "pixel" (we never had real geo coords),
    - src_crs is already WGS84.
    """
    if change_path is None or not change_path.exists():
        return
    if not src_crs:
        return
    src_norm = str(src_crs).strip()
    if not src_norm or src_norm.lower() == "pixel":
        return
    if src_norm.upper() in ("EPSG:4326", "WGS84", "OGC:CRS84"):
        return

    try:
        from pyproj import Transformer
    except Exception as e:
        summary.warn(f"reproject_change_geojson: pyproj unavailable ({e})")
        return

    try:
        transformer = Transformer.from_crs(src_norm, "EPSG:4326", always_xy=True)
    except Exception as e:
        summary.warn(f"reproject_change_geojson: bad src_crs={src_norm} ({e})")
        return

    def _ring(coords: list) -> list:
        out = []
        for pt in coords:
            if not pt or len(pt) < 2:
                continue
            lon, lat = transformer.transform(float(pt[0]), float(pt[1]))
            out.append([lon, lat])
        return out

    def _geom(geom: dict[str, Any]) -> dict[str, Any]:
        t = geom.get("type")
        coords = geom.get("coordinates") or []
        if t == "Polygon":
            return {"type": "Polygon", "coordinates": [_ring(r) for r in coords]}
        if t == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "coordinates": [[_ring(r) for r in poly] for poly in coords],
            }
        return geom

    try:
        payload = json.loads(change_path.read_text())
    except Exception as e:
        summary.warn(f"reproject_change_geojson: read failed ({e})")
        return

    features = payload.get("features") or []
    for feat in features:
        geom = feat.get("geometry")
        if isinstance(geom, dict):
            feat["geometry"] = _geom(geom)
        props = feat.get("properties") or {}
        if "crs" in props and props["crs"] != "EPSG:4326":
            props["source_crs"] = props["crs"]
            props["crs"] = "EPSG:4326"
        feat["properties"] = props

    change_path.write_text(json.dumps(payload, indent=2))


def run_citylens(request: Any, work_dir: Path, progress_cb: ProgressCb = None) -> dict[str, Path]:
    """Orchestrate the Citylens pipeline.

    Progress points: resolve 5, fetch 25, segment 55, refine 65, change 75,
    reconstruct 90, render 95, done 100.

    Produces real artifacts on success.

    On failure, writes only `run_summary.json` with structured error information
    (error_code, error_message, missing_paths) and returns a map containing just
    that summary artifact.
    """

    work_dir = Path(work_dir)
    req = request if isinstance(request, CitylensRequest) else CitylensRequest.model_validate(request)
    summary = PipelineSummary(request=req, work_dir=work_dir, started_at=datetime.utcnow())
    t0 = time.time()
    summary.qa["reference_case_id"] = _slugify_reference_case(req.address)
    summary.qa.setdefault("parity_status", "not_evaluated")

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
        summary.performance["total_runtime_seconds"] = duration_s
        write_run_summary(summary, standard_paths["summary"], extra={"duration_s": duration_s})
        _emit_progress(progress_cb, 100, "done")
        return {"summary": standard_paths["summary"]}

    # Preflight SAM2 assets before any expensive stage work.
    try:
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
            duration = float(time.perf_counter() - t_stage0)
            summary.performance.setdefault("stage_timings_seconds", {})[name] = duration
            summary.stage_status[name] = "ok"
            _log(
                "stage_finished",
                {"stage": name, "duration_secs": duration, "ok": True},
            )
            return out
        except Exception as e:
            duration = float(time.perf_counter() - t_stage0)
            summary.performance.setdefault("stage_timings_seconds", {})[name] = duration
            summary.stage_status[name] = "error"
            _log(
                "stage_finished",
                {
                    "stage": name,
                    "duration_secs": duration,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise

    def _skip_stage(name: str, pct: int, ctx: dict[str, Any]) -> dict[str, Any]:
        _emit_progress(progress_cb, pct, name)
        summary.performance.setdefault("stage_timings_seconds", {})[name] = 0.0
        summary.stage_status[name] = "skipped"
        _log("stage_skipped", {"stage": name, "ok": True})
        return ctx

    def _populate_quality_metrics(ctx: dict[str, Any]) -> None:
        ref_mask = ctx.get("refined_mask", ctx.get("mask"))
        baseline_mask = ctx.get("baseline_footprints_mask")
        if baseline_mask is None:
            baseline_mask = ctx.get("refined_baseline_mask", ctx.get("baseline_mask"))

        summary.qa["baseline_footprints_used"] = bool(ctx.get("baseline_footprints_mask") is not None)
        summary.qa["lidar_used"] = ctx.get("mesh_height_source") == "lidar"

        # Attest to the real input bytes used by this run. These let a reader
        # of run_summary.json answer "did this run actually use real imagery?"
        # without pulling Firestore logs.
        summary.qa["sam2_used"] = bool(req.segmentation_backend == "sam2")
        for key in ("orthophoto_sha256", "baseline_sha256", "lidar_sha256"):
            value = ctx.get(key)
            if isinstance(value, str) and value:
                summary.qa[key] = value

        if ref_mask is not None and baseline_mask is not None:
            try:
                summary.qa["mask_iou"] = mask_iou(ref_mask, baseline_mask)
            except Exception as e:
                summary.warn(f"qa.mask_iou failed: {type(e).__name__}: {e}")
                summary.qa["mask_iou"] = None
        else:
            summary.qa["mask_iou"] = None

        change_path = ctx.get("change_path")
        if change_path is not None and ref_mask is not None and baseline_mask is not None:
            try:
                import json as _json
                from rasterio.transform import Affine
                from rasterio.features import rasterize

                change_transform = ctx.get("orthophoto_transform")
                if change_transform is None:
                    change_transform = Affine.identity()

                # Rasterize ONLY features that describe real change
                # (added / demolished / modified). "unchanged" features
                # are also written to change.geojson for UI purposes but
                # should not count against the change-XOR reference.
                change_types = {"added", "demolished", "modified"}
                gj = _json.loads(Path(change_path).read_text())
                geoms = []
                for feat in gj.get("features") or []:
                    props = feat.get("properties") or {}
                    if str(props.get("change_type", "")).lower() not in change_types:
                        continue
                    geom = feat.get("geometry")
                    if geom:
                        geoms.append((geom, 1))

                out_shape = tuple(np.asarray(ref_mask).shape)  # type: ignore[name-defined]
                if geoms:
                    predicted_change = rasterize(
                        shapes=geoms,
                        out_shape=out_shape,
                        transform=change_transform,
                        fill=0,
                        default_value=1,
                        all_touched=False,
                        dtype="uint8",
                    ).astype(bool)
                else:
                    predicted_change = np.zeros(out_shape, dtype=bool)

                reference_change = np.logical_xor(
                    np.asarray(ref_mask).astype(bool),
                    np.asarray(baseline_mask).astype(bool),
                )
                # HONESTY NOTE: this is NOT an accuracy metric. The
                # "reference" is the raw XOR of the two masks the classifier
                # itself consumes — circular by construction — and the
                # classifier is *designed* to suppress exactly the XOR noise
                # this rewards, so correctness improvements tend to LOWER it.
                # It survives as a coarse mask-agreement/change-magnitude
                # signal only. Per-class accuracy lives in the DOB weak-label
                # harness (research/weak_label_eval.py in citylens-parcel-
                # intel consumers). `change_polygon_f1` is a deprecated alias
                # kept one release for run_summary.json readers.
                xor_f1 = mask_f1(predicted_change, reference_change)
                summary.qa["mask_xor_f1"] = xor_f1
                summary.qa["change_polygon_f1"] = xor_f1
            except Exception as e:
                summary.warn(f"qa.mask_xor_f1 failed: {type(e).__name__}: {e}")
                summary.qa["mask_xor_f1"] = None
                summary.qa["change_polygon_f1"] = None
        else:
            summary.qa["mask_xor_f1"] = None
            summary.qa["change_polygon_f1"] = None

        mesh_footprint_mask = ctx.get("mesh_footprint_mask")
        if mesh_footprint_mask is not None and ref_mask is not None:
            try:
                summary.qa["mesh_footprint_iou"] = mask_iou(mesh_footprint_mask, ref_mask)
            except Exception as e:
                summary.warn(f"qa.mesh_footprint_iou failed: {type(e).__name__}: {e}")
                summary.qa["mesh_footprint_iou"] = None
        else:
            summary.qa["mesh_footprint_iou"] = None

        computed = [
            summary.qa.get("mask_iou") is not None,
            summary.qa.get("mask_xor_f1") is not None,
            summary.qa.get("mesh_footprint_iou") is not None,
        ]
        if all(computed):
            summary.qa["parity_status"] = "complete"
        elif any(computed):
            summary.qa["parity_status"] = "partial"
        else:
            summary.qa["parity_status"] = "not_evaluated"

    ctx: dict[str, Any] = {}
    try:
        ctx = _run_stage("resolve", 5, stage_resolve, ctx)
        ctx = _run_stage("fetch", 25, stage_fetch, ctx)
        ctx = _run_stage("segment", 55, stage_segment, ctx)
        ctx = _run_stage("refine", 65, stage_refine, ctx)

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
    # Honor outputs gating: only require artifacts that were requested.
    required: list[str] = []
    if want_preview:
        required.append("preview")
    if want_change:
        required.append("change")
    if want_mesh:
        required.append("mesh")
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
    summary.performance["total_runtime_seconds"] = duration_s
    _populate_quality_metrics(ctx)
    # GeoJSON RFC 7946 requires WGS84 lng/lat. We write change.geojson in
    # the ortho's CRS so QA can rasterize against orthophoto_transform
    # without re-projection; here we rewrite it in WGS84 so Leaflet/
    # MapboxGL etc. render the polygons on a real map.
    _reproject_change_geojson_to_wgs84(
        standard_paths.get("change"),
        ctx.get("orthophoto_crs"),
        summary,
    )
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

    # Return only paths that exist to avoid callers trying to consume
    # intentionally-skipped artifacts.
    out: dict[str, Path] = {"summary": standard_paths["summary"]}
    for k in ("preview", "change", "mesh"):
        p = standard_paths[k]
        if p.exists():
            out[k] = p
    return out
