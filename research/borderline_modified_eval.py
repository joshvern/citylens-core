"""Evaluate the borderline-modified reclassification fix on the 5 production
demos.

Runs `stage_change` twice per demo against the cached production
`change.geojson` + `orthophoto.tif` inputs:

  baseline:   borderline_margin=0 (pre-fix behavior, current production)
  candidate:  borderline_margin=0.05 (the new default)

Both runs go through the existing `change_algo_improvements_eval.py`
input-recovery helpers, so the diff is purely in the per-class counts +
the new `borderline_modified_reclassifications` qa block.

Usage:
    PYTHONPATH=src:$PYTHONPATH \\
      python research/borderline_modified_eval.py \\
        --runs-dir /tmp/citylens-eval-all \\
        --orthos-dir /tmp/citylens-eval/inputs \\
        --out research/borderline_modified_eval.md

Each per-demo dir under `--runs-dir` must contain preview.png,
change.geojson, run_summary.json (the same artifacts the all-demos eval
expects).
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


# Two configs: pre-fix (margin=0) vs new default (margin=0.05). All other
# improvements (registration, surface delta-E, smoothing, confidence) stay
# at their default settings — we want a clean A/B on the new pass alone.
CONFIGS: list[tuple[str, dict[str, str | None]]] = [
    ("pre-fix (margin=0)", {"CITYLENS_CHANGE_MODIFIED_BORDERLINE_MARGIN": "0"}),
    ("new-default (margin=0.05)", {}),
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
                baseline_path=ortho,  # surface-change uses ortho-as-self → flat Δ-E
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
        "Δ modified",
        "kept_by_surface",
        "unchanged_iou_used",
        "border_band",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        if r.get("skipped"):
            continue
        before = r["rows"][0]["summary"]["qa"].get("change_counts") or {}
        after = r["rows"][1]["summary"]["qa"].get("change_counts") or {}
        recls = (
            r["rows"][1]["summary"]["qa"].get("borderline_modified_reclassifications")
            or {}
        )
        thresh = r["rows"][1]["summary"]["qa"].get("unchanged_iou_used")
        margin = r["rows"][1]["summary"]["qa"].get("borderline_modified_margin", 0.05)
        b_lo = max(0.20, (thresh or 0) - margin)
        m_before = before.get("modified", 0)
        m_after = after.get("modified", 0)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['address'][:32]} (`{r['run_id'][:8]}`)",
                    f"{before.get('unchanged', 0)}/{m_before}/"
                    f"{before.get('demolished', 0)}/{before.get('added', 0)}",
                    f"{after.get('unchanged', 0)}/{m_after}/"
                    f"{after.get('demolished', 0)}/{after.get('added', 0)}",
                    f"{m_before} → {m_after}",
                    str(recls.get("kept_by_surface_change", 0)),
                    f"{thresh}",
                    f"[{b_lo:.3f}, {thresh})" if thresh else "—",
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
    out_lines.append(
        "# Research: borderline-modified reclassification — all 5 demos"
    )
    out_lines.append("")
    out_lines.append(
        "Evaluates the new `_modified_borderline_margin` two-stage classifier "
        "added to `stage_change`. Compares the pre-fix behavior (margin=0) "
        "against the new default (margin=0.05) on every cached production demo "
        "in `gs://citylens-001-artifacts/runs/`."
    )
    out_lines.append("")
    out_lines.append(
        "**Hypothesis**: most production \"modified\" features have IoU clustered "
        "just below the (adaptive) unchanged threshold; demoting that cluster to "
        "unchanged should drop the per-tile modified rate from 10–25 % toward a "
        "more plausible single-digit baseline without losing real change signal "
        "(features deeper into the modified band stay modified)."
    )
    out_lines.append("")

    results: list[dict[str, Any]] = []
    for run_id, address in DEMOS:
        print(f"==> {run_id[:12]}  {address}", flush=True)
        results.append(
            _eval_one_demo(
                run_id=run_id,
                address=address,
                runs_dir=args.runs_dir,
                orthos_dir=args.orthos_dir,
            )
        )

    out_lines.append("## Before / after")
    out_lines.append("")
    out_lines.append(_format_summary_table(results))
    out_lines.append("")

    skipped = [r for r in results if r.get("skipped")]
    if skipped:
        out_lines.append("## Skipped demos")
        out_lines.append("")
        for r in skipped:
            out_lines.append(f"- `{r['run_id'][:8]}` {r['address']} — {r['skipped']}")
        out_lines.append("")

    out_lines.append("## Per-demo detail")
    out_lines.append("")
    for r in results:
        if r.get("skipped"):
            continue
        out_lines.append(f"### {r['address']} (`{r['run_id'][:12]}`)")
        out_lines.append("")
        for row in r["rows"]:
            qa = row["summary"]["qa"]
            out_lines.append(
                f"- `{row['label']}`: "
                f"counts={qa.get('change_counts')} "
                f"unchanged_iou_used={qa.get('unchanged_iou_used')} "
                f"borderline={qa.get('borderline_modified_reclassifications')}"
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
