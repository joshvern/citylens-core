from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..models import CitylensRequest, PipelineSummary


# Palette for change-aware preview. Picked for readability over aerial
# imagery: gray sits back, yellow/red/green pop against roofs/pavement.
_CHANGE_COLORS: dict[str, tuple[int, int, int]] = {
    "unchanged": (140, 140, 140),
    "modified": (255, 200, 0),
    "demolished": (220, 30, 30),
    "added": (0, 200, 60),
}
_FILL_ALPHA = 110  # 0..255 — enough to see the change, not so much we lose imagery
_OUTLINE_ALPHA = 230


def _load_change_features(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    feats = payload.get("features")
    if not isinstance(feats, list):
        return []
    return [f for f in feats if isinstance(f, dict) and f.get("geometry")]


def _paint_change_overlay(
    base_rgba: np.ndarray,
    features: list[dict[str, Any]],
    transform: Any,
) -> np.ndarray:
    """Rasterize each change feature and composite color-coded fills +
    outlines onto base_rgba. Pure NumPy + PIL; no scipy/skimage."""
    from rasterio.features import rasterize

    H, W = base_rgba.shape[:2]
    out = base_rgba.copy()

    # Draw in a deterministic order so adjacent-building colors layer
    # predictably: unchanged first (back), then modified, added, demolished
    # (most important on top).
    order = ["unchanged", "modified", "added", "demolished"]
    by_kind: dict[str, list[dict[str, Any]]] = {k: [] for k in order}
    for feat in features:
        kind = (feat.get("properties") or {}).get("change_type")
        if kind in by_kind:
            by_kind[kind].append(feat)

    for kind in order:
        feats = by_kind[kind]
        if not feats:
            continue
        color = _CHANGE_COLORS[kind]
        try:
            mask = rasterize(
                [(f["geometry"], 1) for f in feats],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
        except Exception:
            continue
        if not mask.any():
            continue

        # Alpha-blend a flat fill onto the covered pixels.
        a = _FILL_ALPHA / 255.0
        for ch, c_val in enumerate(color):
            out[mask, ch] = np.clip(
                out[mask, ch].astype(np.float32) * (1.0 - a) + c_val * a,
                0,
                255,
            ).astype(np.uint8)
        out[mask, 3] = 255

        # Outline: ring = mask XOR eroded-mask (1px). Pure NumPy 4-neighbor erode.
        up = np.zeros_like(mask)
        up[1:, :] = mask[:-1, :]
        dn = np.zeros_like(mask)
        dn[:-1, :] = mask[1:, :]
        lf = np.zeros_like(mask)
        lf[:, 1:] = mask[:, :-1]
        rg = np.zeros_like(mask)
        rg[:, :-1] = mask[:, 1:]
        interior = mask & up & dn & lf & rg
        ring = mask & ~interior
        for ch, c_val in enumerate(color):
            out[ring, ch] = c_val
        out[ring, 3] = _OUTLINE_ALPHA

    return out


def _bake_legend_and_year(
    img: Image.Image,
    counts: dict[str, int] | None,
    imagery_year: int | None,
    baseline_year: int | None,
) -> None:
    """Draw a semi-transparent legend in the lower-left and a year label
    in the upper-right. Mutates `img` in place."""
    W, H = img.size
    draw = ImageDraw.Draw(img, mode="RGBA")

    # Year label — upper right
    if imagery_year is not None:
        label = (
            f"{imagery_year} imagery vs {baseline_year} baseline"
            if baseline_year is not None
            else f"{imagery_year} imagery"
        )
        pad = 8
        tw = len(label) * 7  # rough width for default PIL font
        th = 14
        box = (W - tw - 2 * pad - 10, 10, W - 10, 10 + th + 2 * pad)
        draw.rectangle(box, fill=(0, 0, 0, 160))
        draw.text((box[0] + pad, box[1] + pad), label, fill=(255, 255, 255, 255))

    # Legend — lower left
    rows = []
    for kind in ("unchanged", "modified", "added", "demolished"):
        color = _CHANGE_COLORS[kind]
        count = None
        if counts is not None:
            count = counts.get(kind)
        text = kind if count is None else f"{kind} ({count})"
        rows.append((color, text))

    pad = 8
    swatch = 14
    row_h = swatch + 4
    max_label_w = max(len(t) for _, t in rows) * 7
    panel_w = pad * 3 + swatch + max_label_w
    panel_h = pad * 2 + row_h * len(rows)
    x0 = 10
    y0 = H - panel_h - 10
    draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), fill=(0, 0, 0, 170))

    for i, (color, text) in enumerate(rows):
        ry = y0 + pad + i * row_h
        draw.rectangle(
            (x0 + pad, ry, x0 + pad + swatch, ry + swatch),
            fill=(*color, 255),
            outline=(255, 255, 255, 255),
        )
        draw.text(
            (x0 + pad * 2 + swatch, ry),
            text,
            fill=(255, 255, 255, 255),
        )


def stage_render(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "preview.png"

    ortho_path = Path(ctx.get("orthophoto_path", work_dir / "orthophoto.png"))
    img = Image.open(ortho_path).convert("RGBA")
    base = np.array(img).astype(np.uint8)

    transform = ctx.get("orthophoto_transform")
    change_path = ctx.get("change_path")
    features: list[dict[str, Any]] = []
    if change_path is not None and Path(change_path).exists() and transform is not None:
        features = _load_change_features(Path(change_path))

    if features:
        overlay = _paint_change_overlay(base, features, transform)
        out_img = Image.fromarray(overlay)
        counts = summary.qa.get("change_counts") if isinstance(summary.qa, dict) else None
        _bake_legend_and_year(
            out_img,
            counts if isinstance(counts, dict) else None,
            request.imagery_year,
            request.baseline_year,
        )
        out_img.save(out_path)
        summary.qa["preview_source"] = "change_classified"
        return {**ctx, "preview_path": out_path}

    # Fallback: single red mask over imagery — keeps smoke tests and
    # non-NYC paths working when change.geojson isn't produced.
    mask = ctx.get("mask")
    if mask is None:
        Image.fromarray(base).save(out_path)
        summary.qa["preview_source"] = "ortho_only"
        return {**ctx, "preview_path": out_path}

    m = np.asarray(mask).astype(bool)
    overlay = base.copy()
    overlay[m, 0] = 255
    overlay[m, 1] = 0
    overlay[m, 2] = 0
    overlay[m, 3] = 160
    Image.fromarray(overlay).save(out_path)
    summary.qa["preview_source"] = "mask_red"
    return {**ctx, "preview_path": out_path}
