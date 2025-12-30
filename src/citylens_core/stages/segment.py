from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import CitylensRequest, PipelineSummary
from ..sam.sam2_runner import Sam2UnavailableError, run_sam2_auto_mask


def _placeholder_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _simple_threshold_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = image_rgb.mean(axis=2)
    thr = float(gray.mean())
    return (gray > thr).astype(np.uint8)


def stage_segment(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    ortho_path = Path(ctx.get("orthophoto_path", work_dir / "orthophoto.png"))
    img = Image.open(ortho_path).convert("RGB")
    image_rgb = np.array(img)
    h, w = image_rgb.shape[:2]

    mask_path = work_dir / "mask.png"

    backend = request.segmentation_backend
    if backend in ("unet", "smp"):
        mask = _simple_threshold_mask(image_rgb)
        summary.warn(f"segmentation:{backend}: using placeholder threshold mask")
    elif backend == "sam2":
        try:
            mask = run_sam2_auto_mask(
                image_rgb,
                cfg_path=Path(request.sam2_cfg or ""),
                ckpt_path=Path(request.sam2_checkpoint or ""),
            )
        except Sam2UnavailableError as e:
            summary.warn(f"segmentation:sam2: {e}; using placeholder mask")
            mask = _placeholder_mask(h, w)
        except Exception as e:
            summary.warn(f"segmentation:sam2: {type(e).__name__}: {e}; using placeholder mask")
            mask = _placeholder_mask(h, w)
    else:
        summary.warn(f"segmentation:unknown-backend:{backend}; using placeholder mask")
        mask = _placeholder_mask(h, w)

    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    return {**ctx, "mask": mask, "mask_path": mask_path}
