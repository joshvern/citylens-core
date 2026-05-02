"""Evaluate the wider-baseline-dilation + majority-inside-dilation fix on
the 5 production demos.

Runs `stage_change` twice per demo against cached production
`change.geojson` + `orthophoto.tif` inputs:

  baseline:   pre-fix behavior — DILATE_PX=8, no frac gate
  candidate:  new defaults — DILATE_PX=24, frac=0.5

A/B comparison is per-class counts plus the new
`majority_inside_baseline_dilation` reject reason. Goal: false-positive
"added" counts should drop in dense / mis-registered demos (Brooklyn,
Hudson Yards, East Village) without dropping legitimate new-construction
"added" counts elsewhere.

Usage:
    PYTHONPATH=src:$PYTHONPATH \\
      python research/added_misalignment_eval.py \\
        --runs-dir /tmp/citylens-eval-all \\
        --orthos-dir /tmp/citylens-eval/inputs \\
        --out research/added_misalignment_eval.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import rasterio

from change_algo_improvements_eval import (
    _baseline_features_from_production,
    _crop_ortho_to_data,
    _rasterize_baseline_union,
    _recover_imagery_mask_from_production,
    run_one,
)
from change_algo_improvements_eval_all_demos import (
    DEMOS,
    _demo_centroid_3857,
    _find_covering_ortho,
)


CONFIGS: list[tuple[str, dict[str, str | None]]] = [
    (
        "pre-fix (dilate=8, no frac)",
        {
            "CITYLENS_CHANGE_ADDED_BASELINE_DILATE_PX": "8",
            "CITYLENS_CHANGE_ADDED_MAX_INSIDE_DILATION_FRAC": "1.0",
        },
    ),
    ("new-default (dilate=24, frac=0.5)", {}),
]


def _eval_one_demo(
    *,
    run_id: str,
    address: str,
    runs_dir: Path,
    orthos_dir: Path,
) -> dict[str, Any]:
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
            "skipped": "no cached ortho covers demo centroid",
        }

    ortho = demo_dir / "orthophoto.cropped.tif"
    if not ortho.exists():
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
            "skipped": "rasterized SAM2 proxy mask empty",
        }

    baseline_features = _baseline_features_from_production(change_gj, crs)
    baseline_mask = _rasterize_baseline_union(
        baseline_features, out_shape=ortho_shape, transform=transform
    )

    rows: list[dict[str, Any]] = []
    for label, overrides in CONFIGS:
        rows.append(
            run_one(
                label=label,
                env_overrides=overrides,
                orthophoto_path=ortho,
                baseline_path=ortho,
                baseline_features=baseline_features,
                imagery_mask=imagery_mask,
                baseline_mask=baseline_mask,
                transform=transform,
                crs=crs,
            )
        )
    return {
        "run_id": run_id,
        "address": address,
        "skipped": None,
        "rows": rows,
    }


def _format_summary_table(results: list[dict[str, Any]]) -> str:
    headers = [
        "demo",
        "before u/m/d/a",
        "after u/m/d/a",
        "Δ added",
        "centroid_near",
        "majority_inside",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        if r.get("skipped"):
            continue
        before_qa = r["rows"][0]["summary"]["qa"]
        after_qa = r["rows"][1]["summary"]["qa"]
        before = before_qa.get("change_counts") or {}
        after = after_qa.get("change_counts") or {}
        rej_after = after_qa.get("added_rejected") or {}
        a_before = before.get("added", 0)
        a_after = after.get("added", 0)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['address'][:32]} (`{r['run_id'][:8]}`)",
                    f"{before.get('unchanged', 0)}/{before.get('modified', 0)}/"
                    f"{before.get('demolished', 0)}/{a_before}",
                    f"{after.get('unchanged', 0)}/{after.get('modified', 0)}/"
                    f"{after.get('demolished', 0)}/{a_after}",
                    f"{a_before} → {a_after}",
                    str(rej_after.get("centroid_near_baseline", 0)),
                    str(rej_after.get("majority_inside_baseline_dilation", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


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
    out_lines.append("# Research: 'added' false-positive fix — all 5 demos")
    out_lines.append("")
    out_lines.append(
        "Evaluates the wider near-baseline gate (DILATE_PX 8 → 24) plus "
        "the new majority-inside-dilation guard. Compares pre-fix vs the "
        "new defaults on every cached production demo. Goal: drop "
        "false-positive 'added' counts on dense demos (Brooklyn, Hudson "
        "Yards, East Village) without dropping legitimate new-construction "
        "additions elsewhere."
    )
    out_lines.append("")

    results: list[dict[str, Any]] = []
    for run_id, address in DEMOS:
        print(f"==> {run_id[:12]}  {address}", flush=True)
        result = _eval_one_demo(
            run_id=run_id,
            address=address,
            runs_dir=args.runs_dir,
            orthos_dir=args.orthos_dir,
        )
        results.append(result)

    out_lines.append("## Summary")
    out_lines.append("")
    out_lines.append(_format_summary_table(results))
    out_lines.append("")

    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        out_lines.append("## Skipped demos")
        out_lines.append("")
        for r in skipped:
            out_lines.append(f"- `{r['run_id']}` ({r['address']}): {r['skipped']}")
        out_lines.append("")

    text = "\n".join(out_lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
