"""Vendored (adapted) from ../Urban3D-DeepRecon/src/segmentation.py.

This is intentionally not used by the core pipeline directly; it exists to ease migration
and preserve a familiar entrypoint for experiments.

Dependencies such as torch/sam2 are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def sam2_inference_segmentation(
    satellite_path: str,
    output_path: str,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    checkpoint: str = "weights/sam2.1_hiera_small.pt",
    device: Optional[str] = None,
):
    """Run SAM2 automatic mask generation and save a single combined mask image.

    This mirrors the Urban3D function shape but uses a simplified auto-mask path.
    """

    image = np.array(Image.open(satellite_path).convert("RGB"))

    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except Exception as e:  # pragma: no cover
        raise ImportError("sam2 not installed; install with citylens-core[sam2]") from e

    cfg_path = Path(model_cfg)
    ckpt_path = Path(checkpoint)
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 assets missing (cfg={cfg_path}, ckpt={ckpt_path}). Run `make sam2-assets`."
        )

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    orig_torch_load = torch.load

    def _torch_load(path, *args, **kwargs):
        if kwargs.get("weights_only", False):
            kwargs["weights_only"] = False
        return orig_torch_load(path, *args, **kwargs)

    torch.load = _torch_load
    try:
        sam2_model = build_sam2(str(cfg_path), str(ckpt_path), device=target_device)
    finally:
        torch.load = orig_torch_load

    gen = SAM2AutomaticMaskGenerator(sam2_model)
    masks = gen.generate(image)
    h, w = image.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        seg = m.get("segmentation") if isinstance(m, dict) else m
        if seg is None:
            continue
        combined = np.logical_or(combined, np.asarray(seg, dtype=bool)).astype(np.uint8)

    Image.fromarray(combined * 255).save(output_path)
