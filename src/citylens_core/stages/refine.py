from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from ..models import CitylensRequest, PipelineSummary
from ..io.geo import clean_binary_mask, geojson_crs_hint, load_geojson_mask


def _load_orthophoto_geometry(ctx: dict[str, Any], work_dir: Path) -> tuple[tuple[int, int] | None, Any | None, str | None]:
    ortho_path = Path(ctx.get("orthophoto_path", work_dir / "orthophoto.png"))
    transform = ctx.get("orthophoto_transform")
    crs = ctx.get("orthophoto_crs")

    shape: tuple[int, int] | None = None
    if ortho_path.exists():
        try:
            with rasterio.open(ortho_path) as src:
                shape = (int(src.height), int(src.width))
                if transform is None:
                    transform = src.transform if src.transform is not None else None
                if crs is None and src.crs is not None:
                    crs = src.crs.to_string()
        except Exception:
            shape = None

    if shape is None and "mask" in ctx:
        mask = np.asarray(ctx["mask"])
        if mask.ndim == 2:
            shape = (int(mask.shape[0]), int(mask.shape[1]))

    return shape, transform, crs


def stage_refine(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    """Deterministically refine segmentation masks and optional baseline guidance."""

    mask = ctx.get("mask")
    if mask is None:
        raise RuntimeError("refine stage requires a segmentation mask")

    shape, transform, crs = _load_orthophoto_geometry(ctx, work_dir)
    min_dim = min(int(np.asarray(mask).shape[0]), int(np.asarray(mask).shape[1]))
    morph_radius = 1 if min_dim >= 9 else 0
    min_component_px = max(1 if min_dim < 9 else 4, int(round(np.asarray(mask).size * 0.00001)))
    refined_mask = clean_binary_mask(
        np.asarray(mask).astype(bool),
        open_radius=morph_radius,
        close_radius=morph_radius,
        min_component_px=min_component_px,
    ).astype(np.uint8)

    baseline_mask = ctx.get("baseline_mask")
    refined_baseline_mask = None
    if baseline_mask is not None:
        baseline_shape = np.asarray(baseline_mask)
        baseline_min_dim = min(int(baseline_shape.shape[0]), int(baseline_shape.shape[1]))
        baseline_morph_radius = 1 if baseline_min_dim >= 9 else 0
        baseline_min_component_px = max(
            1 if baseline_min_dim < 9 else 4,
            int(round(baseline_shape.size * 0.00001)),
        )
        refined_baseline_mask = clean_binary_mask(
            baseline_shape.astype(bool),
            open_radius=baseline_morph_radius,
            close_radius=baseline_morph_radius,
            min_component_px=baseline_min_component_px,
        ).astype(np.uint8)

    baseline_footprints_path = Path(work_dir) / "baseline_footprints.geojson"
    baseline_footprints_mask = None
    if shape is not None and baseline_footprints_path.exists():
        crs_hint = geojson_crs_hint(baseline_footprints_path)
        pixel_space = crs_hint == "pixel"
        baseline_footprints_mask = load_geojson_mask(
            baseline_footprints_path,
            out_shape=shape,
            transform=transform,
            pixel_space=pixel_space,
        ).astype(np.uint8)
        baseline_footprints_mask = clean_binary_mask(
            baseline_footprints_mask.astype(bool),
            open_radius=1,
            close_radius=1,
            min_component_px=max(4, int(round(baseline_footprints_mask.size * 0.00001))),
        ).astype(np.uint8)
        refined_baseline_mask = baseline_footprints_mask

    out = {
        **ctx,
        "refined_mask": refined_mask,
        "refined_baseline_mask": refined_baseline_mask,
    }
    if baseline_footprints_mask is not None:
        out["baseline_footprints_mask"] = baseline_footprints_mask
        out["baseline_footprints_path"] = baseline_footprints_path
    if crs is not None:
        out["orthophoto_crs"] = crs
    if transform is not None:
        out["orthophoto_transform"] = transform

    return out
