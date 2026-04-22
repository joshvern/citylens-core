from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..io.geo import geojson_crs_hint, load_geojson_mask
from ..models import CitylensRequest, PipelineSummary
from ..sam.sam2_runner import (
    Sam2UnavailableError,
    run_sam2_auto_mask,
    run_sam2_baseline_prompted,
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

    if mode == "prompted" and not baseline_has_features:
        raise RuntimeError(
            "CITYLENS_SAM2_MODE=prompted but baseline_footprints.geojson is missing "
            "or has no usable features"
        )

    summary.qa["sam2_mode"] = "prompted" if use_prompted else "auto"

    mask = None
    try:
        if use_prompted:
            assert baseline_footprints_mask is not None
            mask = run_sam2_baseline_prompted(
                image_rgb,
                baseline_footprints_mask,
                cfg_path=Path(request.sam2_cfg or ""),
                ckpt_path=Path(request.sam2_checkpoint or ""),
            )
        else:
            mask = run_sam2_auto_mask(
                image_rgb,
                cfg_path=Path(request.sam2_cfg or ""),
                ckpt_path=Path(request.sam2_checkpoint or ""),
            )
    except Sam2UnavailableError as e:
        raise RuntimeError(f"SAM2 unavailable: {e}") from e

    mask_path = work_dir / "mask_imagery.png"
    Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).save(mask_path)

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
        "mask": np.asarray(mask, dtype=np.uint8),
        "mask_path": mask_path,
        "baseline_mask": None if baseline_mask is None else np.asarray(baseline_mask, dtype=np.uint8),
        "baseline_mask_path": baseline_mask_path,
    }
