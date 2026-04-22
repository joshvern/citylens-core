from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .assets import Sam2AssetsMissingError, assets_root, ensure_sam2_assets

_logger = logging.getLogger(__name__)


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


def _build_sam2_model(cfg_path: Path, ckpt_path: Path, device: Optional[str] = None):
    """Validate assets, import sam2, and build a ready-to-use model.

    Centralized so both AMG and prompted paths share the same setup and
    PyTorch-2.6 `weights_only` workaround.
    """
    try:
        ensure_sam2_assets(cfg_path, ckpt_path)
    except Sam2AssetsMissingError as e:
        raise Sam2UnavailableError(str(e)) from e

    cfg = str(cfg_path).strip()
    ckpt = str(_resolve_checkpoint_path(ckpt_path).resolve())

    try:
        import torch
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
        model = build_sam2(cfg, ckpt, device=target_device)
    finally:
        torch.load = orig_torch_load

    return torch, model


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

    torch, model = _build_sam2_model(cfg_path, ckpt_path, device=device)

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:  # pragma: no cover
        raise Sam2UnavailableError(
            "sam2 not installed; install with `pip install -e .[sam2]`"
        ) from e

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


def _label_components(mask: np.ndarray) -> np.ndarray:
    """Return (labels, n) where labels[i,j]==k for cell in component k (1..n).

    Uses scipy.ndimage.label when available (fast, 8-connected), falls back
    to a pure-numpy flood-fill via skimage if not. Both are already in the
    core dependency tree via rasterio's transitive deps.
    """
    try:
        from scipy.ndimage import label

        labels, n = label(mask, structure=np.ones((3, 3), dtype=bool))
        return labels.astype(np.int64), int(n)
    except Exception:
        from skimage.measure import label as sk_label

        labels = sk_label(mask.astype(bool), connectivity=2)
        return labels.astype(np.int64), int(labels.max())


def run_sam2_baseline_prompted(
    image_rgb: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    cfg_path: Path,
    ckpt_path: Path,
    device: Optional[str] = None,
) -> np.ndarray:
    """Segment `image_rgb` using SAM2ImagePredictor with per-building prompts.

    Each connected component in `baseline_mask` becomes one SAM2 prompt:
      - centroid as a positive point
      - axis-aligned bbox as a box prompt
    For each component, SAM2 returns up to 3 candidate masks; we pick the
    one whose IoU with the baseline component is highest (tie-break: SAM2's
    own pred_iou score). All picked masks are OR'd together.

    Use this path when the work_dir has authoritative baseline footprints
    (e.g. a rasterized NYC county GDB). It's dramatically more accurate
    than the AutomaticMaskGenerator shotgun: instead of sampling the whole
    image and hoping salient regions are buildings, it tells SAM2 exactly
    where the buildings are and asks it to trace their 2024 edges.

    Returns a uint8 mask matching `image_rgb.shape[:2]`.
    """
    baseline = np.asarray(baseline_mask).astype(bool)
    h, w = image_rgb.shape[:2]
    if baseline.shape != (h, w):
        raise ValueError(
            f"baseline_mask shape {baseline.shape} does not match image {image_rgb.shape[:2]}"
        )
    if not baseline.any():
        return np.zeros((h, w), dtype=np.uint8)

    torch, model = _build_sam2_model(cfg_path, ckpt_path, device=device)

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:  # pragma: no cover
        raise Sam2UnavailableError(
            "sam2 not installed; install with `pip install -e .[sam2]`"
        ) from e

    min_area = _int_env("CITYLENS_SAM2_PROMPT_MIN_AREA_PX", 40)
    max_components = _int_env("CITYLENS_SAM2_PROMPT_MAX_COMPONENTS", 200)
    box_pad = _int_env("CITYLENS_SAM2_PROMPT_BOX_PAD_PX", 8)

    try:
        predictor = SAM2ImagePredictor(model)
        predictor.set_image(image_rgb)

        labels, n = _label_components(baseline)
        # Collect components sorted by area (biggest first so we hit the
        # most visually important buildings before hitting any budget).
        comps: list[tuple[int, int]] = []  # (area, label_id)
        for comp_id in range(1, n + 1):
            area = int((labels == comp_id).sum())
            if area < min_area:
                continue
            comps.append((area, comp_id))
        comps.sort(reverse=True)
        if len(comps) > max_components:
            comps = comps[:max_components]

        if not comps:
            return np.zeros((h, w), dtype=np.uint8)

        combined = np.zeros((h, w), dtype=bool)
        for _area, comp_id in comps:
            comp = (labels == comp_id)
            ys, xs = np.where(comp)
            y0 = max(0, int(ys.min()) - box_pad)
            y1 = min(h, int(ys.max()) + 1 + box_pad)
            x0 = max(0, int(xs.min()) - box_pad)
            x1 = min(w, int(xs.max()) + 1 + box_pad)

            # Centroid of the component as a positive point prompt.
            cy = float(ys.mean())
            cx = float(xs.mean())
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            box = np.array([x0, y0, x1, y1], dtype=np.float32)

            try:
                masks, scores, _lowres = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
            except Exception as e:
                _logger.warning(
                    "sam2_prompted_predict_failed",
                    extra={"component": int(comp_id), "error": f"{type(e).__name__}: {e}"},
                )
                continue

            if masks is None or len(masks) == 0:
                continue

            # Pick the candidate that best matches this component (highest IoU
            # with the baseline component). Tie-break on SAM2's own score.
            best_idx = 0
            best_iou = -1.0
            best_score = 0.0
            for i in range(len(masks)):
                m_i = np.asarray(masks[i], dtype=bool)
                if m_i.shape != (h, w):
                    continue
                inter = np.logical_and(m_i, comp).sum()
                union = np.logical_or(m_i, comp).sum()
                iou = float(inter) / float(union) if union > 0 else 0.0
                sc = float(scores[i]) if scores is not None and len(scores) > i else 0.0
                if iou > best_iou or (abs(iou - best_iou) < 1e-9 and sc > best_score):
                    best_iou = iou
                    best_score = sc
                    best_idx = i

            picked = np.asarray(masks[best_idx], dtype=bool)
            if picked.shape == (h, w):
                combined = np.logical_or(combined, picked)

        return combined.astype(np.uint8)
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
