"""Run the change-stage improvements eval across multiple demo runs.

Loops over the demo runs that have a cached orthophoto in the
`gs://citylens-001-artifacts/inputs/` cache. Reuses the helpers from
`change_algo_improvements_eval.py` so the per-demo eval logic is the
same; this script just wraps it in a per-demo loop and emits a
side-by-side comparison table.

Usage:
    PYTHONPATH=src:$PYTHONPATH \
      python research/change_algo_improvements_eval_all_demos.py \
        --runs-dir /tmp/citylens-eval-all \
        --orthos-dir /tmp/citylens-eval/inputs \
        --out research/change_algo_improvements_eval_all_demos.md

Each per-demo dir under `--runs-dir` must contain:
    preview.png
    change.geojson
    run_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rasterio
from shapely.geometry import shape as shp_shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from change_algo_improvements_eval import (
    CONFIGS,
    _baseline_features_from_production,
    _crop_ortho_to_data,
    _format_md_table,
    _rasterize_baseline_union,
    _recover_imagery_mask_from_production,
    _shift_mask,
    _stats,
    run_one,
)


# Demo addresses keyed by run_id (mirrors `citylens-engine/deploy/demo_runs.json`).
DEMOS: list[tuple[str, str]] = [
    ("5f079d78d89c4387a9c0ddd5e3507b5e", "100 E 21st St Brooklyn, NY 11226"),
    ("6b3e42cd42d24d00a8aab288a29bde22", "15 Hudson Yards, New York, NY 10001"),
    ("36e628d469fc43c1999af98b8568470c", "20 Cooper Square, New York, NY 10003"),
    ("c0d396fe74784084a7be703102eb5b5a", "5-49 Borden Ave, Long Island City, NY 11101"),
    ("5885e4d33f1342fc9ef58cf0fae84733", "240 Bedford Ave, Brooklyn, NY 11211"),
]


def _demo_centroid_3857(change_geojson: Path) -> tuple[float, float] | None:
    """Centroid of the demo's change.geojson features in EPSG:3857.
    Used to find which cached ortho geographically covers the demo."""
    gj = json.loads(change_geojson.read_text())
    feats = gj.get("features") or []
    if not feats:
        return None
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = [], []
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        try:
            sh = shp_transform(
                transformer.transform, shp_shape(g)
            )
            xs.append(sh.centroid.x)
            ys.append(sh.centroid.y)
        except Exception:
            continue
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _find_covering_ortho(
    centroid: tuple[float, float], orthos_dir: Path
) -> Path | None:
    """Return the cached ortho whose bounds contain the centroid; None if
    no cached ortho covers it. The ortho's hash-named .tif files are
    pre-cropped here (we crop on read)."""
    cx, cy = centroid
    for p in sorted(orthos_dir.glob("*.tif")):
        with rasterio.open(p) as src:
            b = src.bounds
        if b.left <= cx <= b.right and b.bottom <= cy <= b.top:
            return p
    return None


def _eval_one_demo(
    *,
    run_id: str,
    address: str,
    runs_dir: Path,
    orthos_dir: Path,
) -> dict[str, Any] | None:
    """Run the full eval for a single demo. Returns None when no cached
    ortho covers the demo's footprint area."""
    demo_dir = runs_dir / run_id
    change_gj = demo_dir / "change.geojson"
    if not change_gj.exists():
        return {"run_id": run_id, "address": address, "skipped": "no change.geojson"}

    centroid = _demo_centroid_3857(change_gj)
    if centroid is None:
        return {"run_id": run_id, "address": address, "skipped": "empty change.geojson"}

    raw_ortho = _find_covering_ortho(centroid, orthos_dir)
    if raw_ortho is None:
        return {
            "run_id": run_id,
            "address": address,
            "skipped": (
                "no cached ortho covers demo centroid; would need NYS WMS fetch"
            ),
        }

    # Crop to non-black data region (production does this in
    # `_crop_ortho_to_data_coverage`).
    ortho = demo_dir / "orthophoto.cropped.tif"
    _crop_ortho_to_data(raw_ortho, ortho)

    with rasterio.open(ortho) as src:
        transform = src.transform
        ortho_shape = (src.height, src.width)
        crs = src.crs

    imagery_mask = _recover_imagery_mask_from_production(
        change_gj, out_shape=ortho_shape, transform=transform, target_crs=crs
    )
    if imagery_mask.sum() == 0:
        return {
            "run_id": run_id,
            "address": address,
            "skipped": "rasterized SAM2 proxy mask is empty (ortho extent doesn't overlap demo)",
        }

    baseline_features = _baseline_features_from_production(change_gj, crs)
    baseline_mask = _rasterize_baseline_union(
        baseline_features, out_shape=ortho_shape, transform=transform
    )

    # Use the current ortho as both current+baseline for the surface-Δ
    # smoke test (same approach as the single-demo eval).
    surface_baseline_path = ortho

    aligned_rows: list[dict[str, Any]] = []
    for label, overrides in CONFIGS:
        aligned_rows.append(
            run_one(
                label=label,
                env_overrides=overrides,
                orthophoto_path=ortho,
                baseline_path=surface_baseline_path,
                baseline_features=baseline_features,
                imagery_mask=imagery_mask,
                baseline_mask=baseline_mask,
                transform=transform,
                crs=crs,
            )
        )

    shift_rows: list[dict[str, Any]] = []
    for shift in [(2, 3), (4, 0)]:
        shifted = _shift_mask(imagery_mask, *shift)
        for tag, overrides in (
            (f"shifted {shift}: prod-equiv", CONFIGS[0][1]),
            (f"shifted {shift}: new-default", CONFIGS[1][1]),
        ):
            shift_rows.append(
                run_one(
                    label=tag,
                    env_overrides=overrides,
                    orthophoto_path=ortho,
                    baseline_path=surface_baseline_path,
                    baseline_features=baseline_features,
                    imagery_mask=shifted,
                    baseline_mask=baseline_mask,
                    transform=transform,
                    crs=crs,
                )
            )

    return {
        "run_id": run_id,
        "address": address,
        "skipped": None,
        "ortho_path": str(raw_ortho),
        "ortho_shape": list(ortho_shape),
        "imagery_mask_px": int(imagery_mask.sum()),
        "baseline_mask_px": int(baseline_mask.sum()),
        "baseline_feature_count": len(baseline_features),
        "aligned_rows": aligned_rows,
        "shift_rows": shift_rows,
    }


def _summary_row(demo: dict[str, Any]) -> dict[str, Any]:
    """Pull one row of headline metrics per demo for the top-of-report
    side-by-side."""
    aligned = demo["aligned_rows"]
    shift = demo["shift_rows"]
    # ablations[0] = prod-equiv (all OFF), ablations[1] = new-default
    prod = aligned[0]
    new = aligned[1]
    # Find the (2, 3)-shifted prod-equiv and new-default rows
    s23_prod = next(
        r for r in shift if r["label"].startswith("shifted (2, 3)") and "prod" in r["label"]
    )
    s23_new = next(
        r for r in shift if r["label"].startswith("shifted (2, 3)") and "new" in r["label"]
    )
    # too_small with registration on vs off (under shift)
    s23_prod_too_small = (
        s23_prod["summary"]["qa"].get("added_rejected", {}).get("too_small", 0)
    )
    s23_new_too_small = (
        s23_new["summary"]["qa"].get("added_rejected", {}).get("too_small", 0)
    )
    counts = prod["summary"]["qa"].get("change_counts") or {}
    surf_avail = prod["summary"]["qa"].get("surface_change_available")
    vert_max_off = max(
        (
            n
            for r in aligned
            if r["label"].startswith("ablation: smoothing OFF")
            for n in r["vertex_counts"]
        ),
        default=0,
    )
    vert_max_on = max(
        (n for n in new["vertex_counts"]), default=0
    )
    return {
        "run_id": demo["run_id"],
        "address": demo["address"],
        "u": counts.get("unchanged", 0),
        "m": counts.get("modified", 0),
        "d": counts.get("demolished", 0),
        "a": counts.get("added", 0),
        "feat_total": prod["feature_count"],
        "s23_too_small_off": s23_prod_too_small,
        "s23_too_small_on": s23_new_too_small,
        "vert_max_off": vert_max_off,
        "vert_max_on": vert_max_on,
        "surface_avail": bool(surf_avail),
    }


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "demo",
        "feat",
        "u/m/d/a",
        "shift(2,3) too_small off→on",
        "vert_max off→on",
        "surface_avail",
    ]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    f"{r['address'][:35]} (`{r['run_id'][:8]}`)",
                    str(r["feat_total"]),
                    f"{r['u']}/{r['m']}/{r['d']}/{r['a']}",
                    f"{r['s23_too_small_off']} → {r['s23_too_small_on']}",
                    f"{r['vert_max_off']} → {r['vert_max_on']}",
                    "yes" if r["surface_avail"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("/tmp/citylens-eval-all"))
    ap.add_argument(
        "--orthos-dir", type=Path, default=Path("/tmp/citylens-eval/inputs")
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.runs_dir.exists():
        raise SystemExit(f"missing runs dir: {args.runs_dir}")
    if not args.orthos_dir.exists():
        raise SystemExit(f"missing orthos dir: {args.orthos_dir}")

    out_lines: list[str] = []
    out_lines.append("# Research: change-stage improvements — all 5 demos")
    out_lines.append("")
    out_lines.append(
        "Extends `change_algo_improvements_eval.md` (Brooklyn-only) to "
        "every demo in `citylens-engine/deploy/demo_runs.json` that has "
        "a cached orthophoto in `gs://citylens-001-artifacts/inputs/`. "
        "Demos without cached coverage are listed with the reason — "
        "re-fetching from NYS WMS is out of scope here."
    )
    out_lines.append("")

    results: list[dict[str, Any]] = []
    for run_id, address in DEMOS:
        print(f"==> {run_id[:12]}  {address}", flush=True)
        r = _eval_one_demo(
            run_id=run_id,
            address=address,
            runs_dir=args.runs_dir,
            orthos_dir=args.orthos_dir,
        )
        if r is None:
            r = {"run_id": run_id, "address": address, "skipped": "internal error"}
        results.append(r)

    covered = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    out_lines.append("## Summary table — covered demos")
    out_lines.append("")
    if covered:
        summary_rows = [_summary_row(r) for r in covered]
        out_lines.append(_format_summary_table(summary_rows))
    else:
        out_lines.append("_(no covered demos)_")
    out_lines.append("")

    if skipped:
        out_lines.append("## Skipped demos")
        out_lines.append("")
        for r in skipped:
            out_lines.append(
                f"- `{r['run_id'][:8]}` {r['address']} — {r['skipped']}"
            )
        out_lines.append("")

    out_lines.append("## Per-demo detail")
    out_lines.append("")
    for r in covered:
        out_lines.append(
            f"### {r['address']} (`{r['run_id'][:12]}`)"
        )
        out_lines.append("")
        out_lines.append(
            f"- ortho: `{r['ortho_path']}`, cropped shape `{r['ortho_shape']}`"
        )
        out_lines.append(
            f"- imagery mask coverage: {r['imagery_mask_px']} px"
        )
        out_lines.append(
            f"- baseline raster coverage: {r['baseline_mask_px']} px"
        )
        out_lines.append(
            f"- baseline features (with `source_gdb`): {r['baseline_feature_count']}"
        )
        out_lines.append("")
        out_lines.append("**Aligned configs:**")
        out_lines.append("")
        out_lines.append(_format_md_table(r["aligned_rows"]))
        out_lines.append("")
        out_lines.append("**Shift tests:**")
        out_lines.append("")
        out_lines.append(_format_md_table(r["shift_rows"]))
        out_lines.append("")
        # Vertex + confidence stats per row
        for row in r["aligned_rows"][:2] + r["shift_rows"]:
            qa = row["summary"]["qa"]
            v = _stats(row["vertex_counts"])
            ar = qa.get("added_rejected") or {}
            out_lines.append(
                f"- `{row['label']}`: vert max={v['max']:.0f} med={v['med']:.0f} | "
                f"added_rejected.too_small={ar.get('too_small', 0)} | "
                f"reg_applied={qa.get('registration', {}).get('applied')}"
            )
        out_lines.append("")

    text = "\n".join(out_lines) + "\n"
    print()
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
