from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .assets import Sam2AssetsMissingError, assets_root, ensure_sam2_assets


class Sam2UnavailableError(RuntimeError):
    """Raised when SAM2 is not usable (missing package or missing assets)."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_checkpoint_path(p: Path) -> Path:
    root = assets_root()
    return p if p.is_absolute() else (root / p)


def run_sam2_auto_mask(
    image_rgb: np.ndarray,
    *,
    cfg_path: Path,
    ckpt_path: Path,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run SAM2 automatic mask generation.

    Imports `sam2` only when called to keep base installs lightweight.
    """

    # Validate assets first (controlled error)
    try:
        ensure_sam2_assets(cfg_path, ckpt_path)
    except Sam2AssetsMissingError as e:
        raise Sam2UnavailableError(str(e)) from e

    # IMPORTANT: SAM2's Hydra config loader expects the config as a reference within the
    # installed `sam2` package (e.g. "configs/sam2.1/sam2.1_hiera_s.yaml"), not an absolute
    # filesystem path. Only the checkpoint must be a filesystem path.
    cfg = str(cfg_path).strip()
    ckpt = str(_resolve_checkpoint_path(ckpt_path).resolve())

    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except Exception as e:  # pragma: no cover
        raise Sam2UnavailableError(
            "sam2 not installed; install with `pip install -e .[sam2]`"
        ) from e

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(target_device, str) and target_device.startswith("cuda") and not torch.cuda.is_available():
        target_device = "cpu"

    # PyTorch >=2.6 defaults weights_only=True which can break some SAM2 checkpoints.
    orig_torch_load = torch.load

    def _torch_load(path, *args, **kwargs):
        if kwargs.get("weights_only", False):
            kwargs["weights_only"] = False
        return orig_torch_load(path, *args, **kwargs)

    torch.load = _torch_load
    try:
        # SAM2 expects a config path/name; we use the YAML path.
        model = build_sam2(cfg, ckpt, device=target_device)
    finally:
        torch.load = orig_torch_load
    try:
        # Default SAM2 AMG params allocate 10-20GB on CPU inference for a
        # 1024x1024 image — far too aggressive for a Cloud Run Job.
        # These knobs let the deployer dial the memory / quality tradeoff
        # without changing citylens-core code. Env-var defaults below are
        # what SAM2 itself ships with.
        points_per_side = _int_env("CITYLENS_SAM2_POINTS_PER_SIDE", 32)
        points_per_batch = _int_env("CITYLENS_SAM2_POINTS_PER_BATCH", 64)
        pred_iou_thresh = _float_env("CITYLENS_SAM2_PRED_IOU_THRESH", 0.88)
        stability_thresh = _float_env("CITYLENS_SAM2_STABILITY_SCORE_THRESH", 0.95)
        crop_n_layers = _int_env("CITYLENS_SAM2_CROP_N_LAYERS", 0)

        generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_thresh,
            crop_n_layers=crop_n_layers,
        )
        masks = generator.generate(image_rgb)
        h, w = image_rgb.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            seg = m.get("segmentation") if isinstance(m, dict) else m
            if seg is None:
                continue
            combined = np.logical_or(combined, np.asarray(seg, dtype=bool)).astype(np.uint8)
        return combined
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
