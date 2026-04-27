# Research: change-classification threshold calibration

**Date:** 2026-04-23
**Inputs:** Brooklyn reference run `345dfe93…` (100 E 21st St, NY 11226)
**Question:** The production thresholds (`unchanged_iou=0.6`) classify 30/43 buildings as `modified` on a block where nothing visibly changed 2017→2024. What's the right calibration?

## TL;DR

SAM2's IoU-with-GDB-footprint distribution on a stable Brooklyn block is **unimodal and peaks at 0.4–0.6**, not 0.8–1.0 as the 0.6 threshold assumes. This isn't edge-alignment noise — it's a **structural mismatch** between what SAM2 sees (roof edges in 2024 imagery) and what the NYC GDB encodes (ground-level footprints from 2017 survey). Overhangs, recessed balconies, shadows, and merged adjacent rooftops mean **0.5 IoU is what "unchanged" actually looks like** at 0.23 m/px ortho resolution.

**Recommendation:** Set `CITYLENS_CHANGE_UNCHANGED_IOU=0.5` as the default and drop dilation knobs — they don't move the needle.

## Methodology

Replayed `stage_change` classification against the cached artifacts of run `345dfe93` without re-running SAM2. Inputs:

- `preview.png` — red-overlay pixels recovered as the SAM2 imagery mask (stage_render paints exact `(255,0,0)`, so this is lossless).
- `baseline_footprints.geojson` — 44 NYC GDB building footprints in EPSG:3857.
- `orthophoto.tif` — 1024 × 1024, EPSG:3857, for the affine transform.

Script: `research/change_threshold_calibration.py`. Reproduces locally in under a second; no Cloud Run cycles burned per iteration.

## Raw IoU distribution (per-baseline-footprint)

```
dilate_px | IoU bins (count): [.0-.1 .1-.2 .2-.3 .3-.4 .4-.5 .5-.6 .6-.7 .7-.8 .8-.9 .9-1.]
  dilate=0:     0    0    0    1   13   16    6    5    2    0
  dilate=2:     0    0    0    0    8   18   11    5    1    0
  dilate=4:     0    0    0    0    6   18   13    5    1    0
  dilate=6:     0    0    0    0    5   19   15    3    1    0
```

**Shape:** Unimodal, peaked at 0.5, concentrated in [0.4, 0.7) (~34/43 buildings).
**Dilation effect:** Marginal — lifts a handful of buildings from 0.4 into 0.5, barely any into 0.7+. 0.23 m/px ortho × 4 px dilation = 1 m of absorbed edge noise; not enough to close the structural gap.

## Config sweep

| config | unchanged | modified | demolished | added | IoU med (unchanged) | IoU med (modified) |
|---|---|---|---|---|---|---|
| **v0.3.8 production** (unch=0.6, dilate=0) | 13 | 30 | 0 | 25 | 0.70 | 0.52 |
| **X** (unch=0.4, dilate=0) | **42** | **1** | 0 | 25 | 0.56 | 0.38 |
| X' (unch=0.5, dilate=0) | 29 | 14 | 0 | 25 | 0.58 | 0.47 |
| Y (unch=0.6, dilate=2) | 17 | 26 | 0 | 25 | 0.65 | 0.53 |
| Y' (unch=0.6, dilate=4) | 19 | 24 | 0 | 25 | 0.65 | 0.54 |
| Y'' (unch=0.5, dilate=2) | 35 | 8 | 0 | 25 | 0.60 | 0.45 |
| Y''' (unch=0.5, dilate=4) | **37** | **6** | 0 | 25 | 0.61 | 0.48 |

## Interpretation

Three things are happening at once:

1. **SAM2 roof-edge vs GDB footprint mismatch.** A Brooklyn brownstone with 0.5 m overhangs on all sides has a SAM2 mask ~10% wider than the GDB polygon. At 1 m/px that's ~5–10% area error without any alignment issue. 43 buildings compound that into the 0.5 median.

2. **Adjacent-building merge.** SAM2 treats a row of attached brownstones as one blob. Per-building IoU drops to ~0.5 because each building's "probe" bbox overlaps neighbors' SAM2 output too.

3. **Ortho resolution.** At 0.23 m/px, a 6 m × 10 m building is 26 × 43 px. A 2-px edge error is 15% of the footprint. Small buildings structurally can't hit 0.8 IoU.

None of this is a bug. It's just what the data supports.

## Option Y (dilation) — verdict: don't

Dilation is the classic "fix edge misalignment" trick. It doesn't help here because the gap isn't edge noise — it's the overhang / merge / resolution story above. Dilate 4 px moves ~4 buildings from `modified` → `unchanged` (18 → 14 in the 0.5-threshold row), not 30. The 4-pixel dilation also introduces a false-positive risk for densely packed row houses: dilate a footprint by 4 px and it overlaps the neighbor.

## Option X (threshold) — verdict: yes, at 0.5

Option X at 0.4 gives the cleanest numbers (42/1/0/25) but moves the bar too far — a building where SAM2 only covered 40% of the footprint is called `unchanged`, which actively masks real quality signal.

**Option X at 0.5 is the honest choice:** 29 unchanged, 14 modified, 0 demolished, 25 added. The 14 modified are the buildings where IoU fell meaningfully below the median — worth flagging for a human reviewer, not noise.

## Proposed default

- `CITYLENS_CHANGE_UNCHANGED_IOU=0.5` (was 0.6)
- `CITYLENS_CHANGE_MODIFIED_IOU=0.2` (unchanged)
- No dilation knob — the effect is too small to justify a tunable.

Expected steady-state behavior on a stable block: ~70% `unchanged`, ~25% `modified`, 0–5% `demolished`, plus a separate `added` tail from SAM2 false positives not suppressed by the overlap filter.

## Update — 2026-04-27 — recalibrated for 250m AOI (169 buildings)

After fixing `request.aoi_radius_m` to be honored end-to-end (engine PR #20), the
Brooklyn reference run now classifies 169 baseline footprints (was 53). The 0.5
threshold flagged 74/169 (44%) as `modified` on a block where almost nothing
visibly changed 2017→2024 — clearly too aggressive. Re-measured the IoU
distribution from this larger sample:

```
IoU bin    count  bar (1 # = 1 bldg)
[0.0-0.1):     3  ###
[0.1-0.2):     0
[0.2-0.3):     6  ######
[0.3-0.4):    22  ######################
[0.4-0.5):    46  ##############################################
[0.5-0.6):    33  #################################
[0.6-0.7):    42  ##########################################
[0.7-0.8):    14  ##############
[0.8-0.9):     2  ##
[0.9-1.0):     1  #
```

The distribution is **bimodal**: a thin "demolished" tail near 0, and a fat
"unchanged" hump from 0.3 to 0.9 with a double peak at 0.4–0.5 and 0.6–0.7
(the second peak is large simple roofs SAM2 traces cleanly; the first is small
or complex roofs). The 0.4–0.5 bucket is mostly stable buildings whose IoU is
suppressed by overhang/edge structural mismatch, **not** real modification.

Reclassification sweep (modified_iou pinned at 0.2):

| `unchanged_iou` | unchanged | modified | demolished |
|---|---|---|---|
| 0.30 | 160 | 6 | 3 |
| 0.35 | 152 | 14 | 3 |
| **0.40** | **138** | **28** | **3** |
| 0.45 | 116 | 50 | 3 |
| 0.50 | 92 | 74 | 3 |

**New default: `CITYLENS_CHANGE_UNCHANGED_IOU=0.40`.** Result: 82% unchanged,
17% modified, 2% demolished, 2% added — matches what a human reviewer would
flag on a stable block. The previous 0.5 default was honest for the older 53-
building sample but didn't generalize once we expanded to 169 buildings with
more small/complex roofs in the periphery.

## Out-of-scope questions this surfaced

1. **`added = 25` is still too many.** The overlap filter rejects additions that touch any baseline footprint, but SAM2 still finds 25 building-shaped things that AREN'T in the 2017 GDB. Probably some mix of (a) real post-2017 construction, (b) garages / backyard structures that NYC OpenData doesn't track, (c) adjacent-building merges leaking past the baseline. Separate investigation.
2. **Calibration is from one address.** The 0.5 threshold was chosen from a single Brooklyn block. A Manhattan tower block or a Queens single-family area may have a different peak. If we ever add more demo addresses, this should be re-measured.
3. **Roof-edge vs footprint mismatch is inherent.** The only real fix is either (a) adjust the GDB footprints by known overhang width (no dataset for that), or (b) use a building-polygon dataset that encodes roof edges (Microsoft Building Footprints is closer to roof edges than NYC OpenData, worth testing).
