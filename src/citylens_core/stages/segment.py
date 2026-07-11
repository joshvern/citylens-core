from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..io.geo import binary_mask_stats, geojson_crs_hint, load_geojson_mask
from ..models import CitylensRequest, PipelineSummary
from ..sam.sam2_runner import (
    Sam2UnavailableError,
    run_sam2_auto_mask,
    run_sam2_baseline_prompted,
    run_sam2_prompted_with_discovery,
)

_logger = logging.getLogger(__name__)


def _resolve_sam2_mode() -> str:
    """Return one of: 'auto_fallback' (default), 'prompted', 'auto'.

    - 'auto_fallback': use prompted SAM2 when baseline_footprints.geojson is
      available with usable features; otherwise fall back to AutomaticMaskGenerator.
    - 'prompted': always attempt prompted; fail if no baseline footprints.
    - 'auto': always use AutomaticMaskGenerator (historical behavior).
    """
    raw = os.getenv("CITYLENS_SAM2_MODE", "auto_fallback").strip().lower()
    if raw in ("prompted", "auto", "auto_fallback"):
        return raw
    return "auto_fallback"


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var without letting typos change the default."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _added_discovery_enabled() -> bool:
    """Whether prompted change runs also discover current-only structures.

    A baseline-prompted mask can only segment where a baseline footprint
    supplied a prompt.  It therefore cannot contain a genuinely new building.
    Change outputs default this current-image automatic discovery pass on; the
    switch exists for constrained environments and controlled ablations.
    """
    return _bool_env("CITYLENS_SAM2_ADDED_DISCOVERY", True)


def _has_semantic_current_footprints(
    *, work_dir: Path, ortho_transform: Any | None
) -> bool:
    """Whether a staged current-footprint collection can replace AMG.

    The stager guarantees geometries are already in the orthophoto CRS, so a
    raster transform is the only additional requirement.  Invalid files do
    not suppress automatic discovery; change detection retains its existing
    local fallback instead.
    """
    path = work_dir / "current_footprints.geojson"
    if ortho_transform is None or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("type") == "FeatureCollection"
        and isinstance(payload.get("features"), list)
    )


def _load_baseline_footprints_mask(
    *, work_dir: Path, ortho_shape: tuple[int, int], ortho_transform: Any | None
) -> np.ndarray | None:
    """Rasterize baseline_footprints.geojson to the ortho's pixel grid."""
    gj_path = work_dir / "baseline_footprints.geojson"
    if not gj_path.exists():
        return None
    try:
        crs_hint = geojson_crs_hint(gj_path)
        pixel_space = crs_hint == "pixel"
        mask = load_geojson_mask(
            gj_path,
            out_shape=ortho_shape,
            transform=ortho_transform,
            pixel_space=pixel_space,
        )
        return mask.astype(np.uint8)
    except Exception as e:
        _logger.warning(
            "baseline_footprints_rasterize_failed",
            extra={"error": f"{type(e).__name__}: {e}"},
        )
        return None


def stage_segment(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    ortho_path = Path(ctx.get("orthophoto_path", work_dir / "orthophoto.png"))

    outputs = {str(o).strip().lower() for o in (request.outputs or []) if str(o).strip()}
    want_change = "change" in outputs

    img = Image.open(ortho_path).convert("RGB")
    image_rgb = np.array(img)
    ortho_shape = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))

    # Decide which SAM2 path to run. Prefer prompted when baseline footprints
    # are available — the AMG shotgun segments every salient region (trees,
    # roads, shadows) which is useless for a building-footprint comparison.
    mode = _resolve_sam2_mode()
    baseline_footprints_mask = _load_baseline_footprints_mask(
        work_dir=work_dir,
        ortho_shape=ortho_shape,
        ortho_transform=ctx.get("orthophoto_transform"),
    )
    baseline_has_features = baseline_footprints_mask is not None and bool(
        baseline_footprints_mask.any()
    )

    use_prompted = (mode == "prompted") or (
        mode == "auto_fallback" and baseline_has_features
    )
    semantic_current_available = want_change and _has_semantic_current_footprints(
        work_dir=work_dir,
        ortho_transform=ctx.get("orthophoto_transform"),
    )
    summary.qa["current_footprints_staged"] = bool(
        (work_dir / "current_footprints.geojson").exists()
    )
    summary.qa["current_footprints_semantic_available"] = bool(
        semantic_current_available
    )

    if mode == "prompted" and not baseline_has_features:
        raise RuntimeError(
            "CITYLENS_SAM2_MODE=prompted but baseline_footprints.geojson is missing "
            "or has no usable features"
        )

    summary.qa["sam2_mode"] = "prompted" if use_prompted else "auto"

    mask = None
    added_discovery_mask = None
    added_discovery_mask_path: Path | None = None
    try:
        if use_prompted:
            assert baseline_footprints_mask is not None
            # Prompted SAM2 is intentionally constrained to baseline
            # footprints. Keep that high-precision mask for imagery QA and,
            # when no semantic current layer exists, baseline IoU. A separate
            # current-image AMG pass then finds structures that did not exist
            # at baseline. Never union the two here: doing so would let AMG
            # noise change unchanged/modified/demolished calls.
            if semantic_current_available:
                # The staged vector collection is a complete, semantic
                # current-building source.  Keep prompted SAM for its normal
                # imagery QA, but avoid an unnecessary/noisier AMG pass.
                mask = run_sam2_baseline_prompted(
                    image_rgb,
                    baseline_footprints_mask,
                    cfg_path=Path(request.sam2_cfg or ""),
                    ckpt_path=Path(request.sam2_checkpoint or ""),
                )
                summary.qa["sam2_added_discovery_mode"] = "current_footprints"
                summary.qa["sam2_added_discovery_status"] = "not_needed"
            elif want_change and _added_discovery_enabled():
                summary.qa["sam2_added_discovery_mode"] = "automatic"
                summary.qa["sam2_added_discovery_status"] = "running"
                try:
                    mask, added_discovery_mask = run_sam2_prompted_with_discovery(
                        image_rgb,
                        baseline_footprints_mask,
                        cfg_path=Path(request.sam2_cfg or ""),
                        ckpt_path=Path(request.sam2_checkpoint or ""),
                    )
                except Exception:
                    # Discovery is part of the requested change product.  Do
                    # not silently publish a one-sided "ok" run that cannot
                    # discover additions; the pipeline records this stage as
                    # failed.  Operators can explicitly disable the pass for
                    # a legacy/ablation run.
                    summary.qa["sam2_added_discovery_status"] = "failed"
                    raise
                summary.qa["sam2_added_discovery_status"] = "ok"
            else:
                mask = run_sam2_baseline_prompted(
                    image_rgb,
                    baseline_footprints_mask,
                    cfg_path=Path(request.sam2_cfg or ""),
                    ckpt_path=Path(request.sam2_checkpoint or ""),
                )
                summary.qa["sam2_added_discovery_mode"] = (
                    "disabled" if want_change else "not_requested"
                )
                summary.qa["sam2_added_discovery_status"] = (
                    "disabled" if want_change else "not_requested"
                )
        else:
            mask = run_sam2_auto_mask(
                image_rgb,
                cfg_path=Path(request.sam2_cfg or ""),
                ckpt_path=Path(request.sam2_checkpoint or ""),
            )
            if want_change:
                if semantic_current_available:
                    summary.qa["sam2_added_discovery_mode"] = "current_footprints"
                    summary.qa["sam2_added_discovery_status"] = "not_needed"
                else:
                    # The primary mask is already automatic, so it is also
                    # the discovery mask. Reuse it rather than running AMG
                    # twice.
                    added_discovery_mask = mask
                    summary.qa["sam2_added_discovery_mode"] = "primary_auto"
                    summary.qa["sam2_added_discovery_status"] = "reused"
            else:
                summary.qa["sam2_added_discovery_mode"] = "not_requested"
                summary.qa["sam2_added_discovery_status"] = "not_requested"
    except Sam2UnavailableError as e:
        raise RuntimeError(f"SAM2 unavailable: {e}") from e

    mask_array = np.asarray(mask, dtype=np.uint8)
    mask_path = work_dir / "mask_imagery.png"
    Image.fromarray(mask_array * 255).save(mask_path)

    added_discovery_mask_array: np.ndarray | None = None
    if added_discovery_mask is not None:
        # Preserve object identity for primary-auto mode so refine can reuse
        # the already-cleaned primary mask without duplicate morphology.
        added_discovery_mask_array = (
            mask_array
            if added_discovery_mask is mask
            else np.asarray(added_discovery_mask, dtype=np.uint8)
        )
        if added_discovery_mask_array is mask_array:
            added_discovery_mask_path = mask_path
        else:
            added_discovery_mask_path = work_dir / "mask_added_discovery.png"
            Image.fromarray(added_discovery_mask_array * 255).save(
                added_discovery_mask_path
            )
        summary.qa["sam2_added_discovery_raw"] = binary_mask_stats(
            added_discovery_mask_array
        )

    # For the baseline reference: if we rasterized the footprints, use that
    # directly — it's the authoritative 2017 ground truth. Running SAM2 on a
    # rasterized footprints image is circular and loses information.
    baseline_mask: np.ndarray | None = None
    baseline_mask_path: Path | None = None
    if want_change:
        if baseline_footprints_mask is not None:
            baseline_mask = baseline_footprints_mask
            baseline_mask_path = work_dir / "mask_baseline.png"
            Image.fromarray((baseline_mask * 255).astype(np.uint8)).save(baseline_mask_path)
        else:
            baseline_path = Path(ctx.get("baseline_path", work_dir / "baseline.png"))
            baseline_img = Image.open(baseline_path).convert("RGB")
            baseline_rgb = np.array(baseline_img)
            try:
                baseline_mask = run_sam2_auto_mask(
                    baseline_rgb,
                    cfg_path=Path(request.sam2_cfg or ""),
                    ckpt_path=Path(request.sam2_checkpoint or ""),
                )
            except Sam2UnavailableError as e:
                raise RuntimeError(f"SAM2 unavailable (baseline): {e}") from e
            baseline_mask_path = work_dir / "mask_baseline.png"
            Image.fromarray((np.asarray(baseline_mask, dtype=np.uint8) * 255)).save(
                baseline_mask_path
            )

    return {
        **ctx,
        "mask": mask_array,
        "mask_path": mask_path,
        "added_discovery_mask": added_discovery_mask_array,
        "added_discovery_mask_path": added_discovery_mask_path,
        "baseline_mask": None if baseline_mask is None else np.asarray(baseline_mask, dtype=np.uint8),
        "baseline_mask_path": baseline_mask_path,
    }
