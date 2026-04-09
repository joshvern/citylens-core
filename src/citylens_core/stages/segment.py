from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import CitylensRequest, PipelineSummary
from ..sam.sam2_runner import Sam2UnavailableError, run_sam2_auto_mask


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
    mask = None
    try:
        mask = run_sam2_auto_mask(
            image_rgb,
            cfg_path=Path(request.sam2_cfg or ""),
            ckpt_path=Path(request.sam2_checkpoint or ""),
        )
    except Sam2UnavailableError as e:
        raise RuntimeError(f"SAM2 unavailable: {e}") from e

    mask_path = work_dir / "mask_imagery.png"
    Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).save(mask_path)

    baseline_mask = None
    baseline_mask_path = None
    if want_change:
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
        Image.fromarray((np.asarray(baseline_mask, dtype=np.uint8) * 255)).save(baseline_mask_path)

    return {
        **ctx,
        "mask": np.asarray(mask, dtype=np.uint8),
        "mask_path": mask_path,
        "baseline_mask": None if baseline_mask is None else np.asarray(baseline_mask, dtype=np.uint8),
        "baseline_mask_path": baseline_mask_path,
    }
