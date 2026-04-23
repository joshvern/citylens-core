"""Calibrate change-classification thresholds against a real demo run.

Run locally against cached inputs from the Brooklyn reference run. Doesn't
re-execute SAM2 — instead it recovers the SAM2 imagery mask from the red
channel of preview.png (which stage_render paints with exact (255,0,0)).

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python research/change_threshold_calibration.py \\
        --inputs /tmp/calib \\
        --out research/change_threshold_calibration_results.md

Expects the inputs dir to contain:
    preview.png
    baseline_footprints.geojson  (reprojected to ortho CRS by the worker)
    orthophoto.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize


# ----------------------------------------------------------------------
# Input recovery
# ----------------------------------------------------------------------


def _recover_imagery_mask(preview_path: Path) -> np.ndarray:
    """SAM2 mask is wherever the red overlay is exactly (255, 0, 0)."""
    im = np.array(Image.open(preview_path).convert("RGBA"))
    return (
        (im[..., 0] == 255)
        & (im[..., 1] == 0)
        & (im[..., 2] == 0)
    ).astype(bool)


def _rasterize_baseline(
    geojson_path: Path, *, out_shape: tuple[int, int], transform: Any
) -> np.ndarray:
    gj = json.loads(geojson_path.read_text())
    shapes = [(f["geometry"], 1) for f in gj.get("features", []) if f.get("geometry")]
    if not shapes:
        return np.zeros(out_shape, dtype=bool)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype="uint8",
    )
    return mask.astype(bool)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Pure-numpy binary dilation (Chebyshev ball). Radius=0 is identity."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


# ----------------------------------------------------------------------
# Classification (matches stage_change semantics at the per-source level)
# ----------------------------------------------------------------------


def classify(
    *,
    imagery_mask: np.ndarray,
    baseline_source_features: list[dict],
    baseline_mask_union: np.ndarray,
    transform: Any,
    ortho_shape: tuple[int, int],
    unchanged_iou: float,
    modified_iou: float,
    dilate_baseline_px: int,
    min_area_px_added: int,
    added_overlap_cap: float,
) -> dict:
    counts = {"unchanged": 0, "modified": 0, "demolished": 0, "added": 0}
    ious: dict[str, list[float]] = {"unchanged": [], "modified": [], "demolished": []}

    # Per-source-feature classification with optional baseline dilation.
    for feat in baseline_source_features:
        geom = feat.get("geometry")
        if not geom:
            continue
        single = rasterize(
            shapes=[(geom, 1)],
            out_shape=ortho_shape,
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        if not single.any():
            continue

        # Option Y: dilate the footprint before IoU so SAM2's edge noise
        # doesn't penalize small buildings.
        probe = _dilate_mask(single, dilate_baseline_px)

        ys, xs = np.where(probe)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad_y = max(1, (y1 - y0) // 10)
        pad_x = max(1, (x1 - x0) // 10)
        y0 = max(0, y0 - pad_y)
        x0 = max(0, x0 - pad_x)
        y1 = min(probe.shape[0], y1 + pad_y)
        x1 = min(probe.shape[1], x1 + pad_x)

        fp_roi = probe[y0:y1, x0:x1]
        im_roi = imagery_mask[y0:y1, x0:x1]
        inter = int(np.logical_and(fp_roi, im_roi).sum())
        union = int(np.logical_or(fp_roi, im_roi).sum())
        iou = float(inter) / float(union) if union > 0 else 0.0

        if iou >= unchanged_iou:
            t = "unchanged"
        elif iou >= modified_iou:
            t = "modified"
        else:
            t = "demolished"
        counts[t] += 1
        ious[t].append(iou)

    # 'Added' detection — same pass regardless of config.
    added_px = np.logical_and(imagery_mask, np.logical_not(baseline_mask_union))
    # A fast-and-cheap connected-component labeler.
    from rasterio.features import shapes

    for geom, value in shapes(added_px.astype("uint8"), mask=added_px, transform=transform):
        if int(value) != 1:
            continue
        comp_mask = rasterize(
            shapes=[(geom, 1)],
            out_shape=ortho_shape,
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=False,
            dtype="uint8",
        ).astype(bool)
        area_px = int(comp_mask.sum())
        if area_px < min_area_px_added:
            continue
        overlap_touch = int(
            np.logical_and(comp_mask, _dilate_mask(baseline_mask_union, 1)).sum()
        )
        if (overlap_touch / area_px) > added_overlap_cap:
            continue
        counts["added"] += 1

    return {"counts": counts, "ious": ious}


def summarize(result: dict) -> str:
    c = result["counts"]
    ious = result["ious"]
    total_baseline = c["unchanged"] + c["modified"] + c["demolished"]
    parts = [
        f"unchanged={c['unchanged']:3d}",
        f"modified={c['modified']:3d}",
        f"demolished={c['demolished']:3d}",
        f"added={c['added']:3d}",
        f"baseline_total={total_baseline}",
    ]
    for t in ("unchanged", "modified"):
        xs = ious.get(t) or []
        if xs:
            parts.append(
                f"{t}_iou_med={sorted(xs)[len(xs)//2]:.2f}"
            )
    return " ".join(parts)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("/tmp/calib"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = args.inputs
    preview = d / "preview.png"
    baseline_gj = d / "baseline_footprints.geojson"
    ortho = d / "orthophoto.tif"
    for p in (preview, baseline_gj, ortho):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    with rasterio.open(ortho) as src:
        transform = src.transform
        ortho_shape = (src.height, src.width)
        crs = src.crs

    imagery_mask = _recover_imagery_mask(preview)
    src_features = json.loads(baseline_gj.read_text()).get("features") or []
    baseline_union = _rasterize_baseline(
        baseline_gj, out_shape=ortho_shape, transform=transform
    )

    print(f"# Research: change classification threshold calibration")
    print()
    print(f"Inputs:")
    print(f"  ortho: {ortho_shape[0]}x{ortho_shape[1]}, crs={crs}")
    print(f"  imagery mask coverage: {imagery_mask.sum()} px ({100*imagery_mask.sum()/imagery_mask.size:.1f}%)")
    print(f"  baseline features: {len(src_features)}")
    print(f"  baseline raster coverage: {baseline_union.sum()} px ({100*baseline_union.sum()/baseline_union.size:.1f}%)")
    print()

    # Configs to sweep.
    # Each row: (label, unchanged_iou, modified_iou, dilate_px)
    configs = [
        ("baseline v0.3.8 (production)",     0.60, 0.20, 0),
        ("Option X: lower unchanged to 0.4", 0.40, 0.20, 0),
        ("Option X': lower to 0.5",          0.50, 0.20, 0),
        ("Option Y: dilate 2px",             0.60, 0.20, 2),
        ("Option Y': dilate 4px",            0.60, 0.20, 4),
        ("Option Y'': dilate 2px + unch 0.5",0.50, 0.20, 2),
        ("Option Y''': dilate 4px + unch 0.5",0.50, 0.20, 4),
    ]

    print(f"| config | unchanged | modified | demolished | added | unchanged IoU med | modified IoU med |")
    print(f"|---|---|---|---|---|---|---|")
    for label, unch, modf, dil in configs:
        r = classify(
            imagery_mask=imagery_mask,
            baseline_source_features=src_features,
            baseline_mask_union=baseline_union,
            transform=transform,
            ortho_shape=ortho_shape,
            unchanged_iou=unch,
            modified_iou=modf,
            dilate_baseline_px=dil,
            min_area_px_added=100,
            added_overlap_cap=0.1,
        )
        c = r["counts"]
        ious = r["ious"]
        unch_med = f"{sorted(ious['unchanged'])[len(ious['unchanged'])//2]:.2f}" if ious['unchanged'] else "—"
        modf_med = f"{sorted(ious['modified'])[len(ious['modified'])//2]:.2f}" if ious['modified'] else "—"
        print(f"| {label} | {c['unchanged']} | {c['modified']} | {c['demolished']} | {c['added']} | {unch_med} | {modf_med} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
