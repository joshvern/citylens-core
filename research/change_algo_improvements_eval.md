# Research: change-stage algorithm improvements eval

Source: cached Brooklyn reference run `5f079d78d89c4387a9c0ddd5e3507b5e` (100 E 21st St Brooklyn, NY 11226), orthophoto pulled from GCS `inputs/` cache.

## Findings

1. **Registration (#1)**: phase-correlation correctly identifies synthetic shifts of (+2,+3), (+4,0) and applies them; defensive guards correctly REJECT a noisy partial recovery on a (0,-5) shift (detected as (-1,+1), IoU gain too small to apply). On already-aligned data the shift detector returns confidence=1.0 but correctly does NOT apply because IoU gain is zero. No regression.
2. **Surface-change Δ-E (#2)**: when given the same ortho twice (zero-truth case), surface_delta_e produces 134/134 samples and fires `surface_changed` on 0 footprints — no false positives. The production pipeline today feeds a 1-band footprint mask as the baseline, which `_load_rgb` correctly rejects (`surface_change_available: false`), so the new code is dormant in prod until a true 2017 RGB ortho is wired up.
3. **Polygon smoothing (#4)**: on production GDB geometry (median 9 vertices), Douglas-Peucker tolerance=0.5 px barely moves the median — the win is on outliers. Max vertex count drops 161 → 147 on aligned data, and 267 → 147 on shifted data where misalignment otherwise produces noisier mask traces.
4. **Classification confidence (#3)**: distribution stays in [0.62, 1.00] (median 0.84) on aligned data and shifts down to [0.60, 1.00] (median 0.76) under (+2,+3) misalignment — the score correctly tracks IoU degradation. Useful as a UI signal.
5. **candidate_added on no-LiDAR-coverage (#5)**: not exercisable in this eval (no LiDAR loaded). Behavior is verified by `tests/test_change_polygons.py::test_lidar_no_coverage_emits_candidate_added_with_low_confidence`. What this eval DOES show is the upstream effect of registration on the candidate-added pipeline: with registration ON the `added_rejected.too_small` counter drops from 204 → 0 on a (+2,+3) shift and 794 → 0 on a (+4,0) shift, because the alignment-corrected mask no longer leaves slivers that get rejected as small candidate-added components.

**Bottom line:** all 5 improvements are integrated correctly. Registration is the most impactful for misregistered data, smoothing trims polygon-vertex tail outliers, confidence produces a useful per-feature score, and surface-Δ-E is dormant in prod until a true 2017 RGB baseline is fetched.

## Inputs

- ortho: `1024x703`, CRS `EPSG:3857`, transform `| 0.49, 0.00,-8233329.96|
| 0.00,-0.49, 4961379.29|
| 0.00, 0.00, 1.00|`
- imagery (SAM2) mask coverage: 345086 px (47.9%)
- baseline features recovered from production `change.geojson`: 134
- baseline raster coverage: 330536 px (45.9%)
- second Brooklyn ortho for surface-Δ probe: `/tmp/citylens-eval/baseline_ortho.cropped.tif`
- LiDAR heights: not loaded for this eval (LiDAR-rescue + candidate_added-on-no-coverage paths are unit-tested)

## Comparison: aligned proxy mask

All ablations against the proxy SAM2 mask derived from production geometry. The mask is by construction well-aligned with the baseline footprints, so registration correctly does NOT fire — this validates the defensive guards (high confidence + low IoU gain) prevent spurious shifts on already-aligned data.

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production-equiv (all OFF) | 134 | 0 | 0 | 2 | 136 | 9 | 15.3 | 0.84 | 0 | 0.0 | 0.0 | False |
| new-default (all ON) | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: registration OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: smoothing OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.3 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: surface delta-E OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |

## Comparison: synthetically-shifted mask (registration test)

The proxy SAM2 mask is shifted by various `(dy, dx)` to simulate the inter-acquisition registration error typically present between NYS 2024 orthos and the 2017 baseline. With registration ON, stage_change should detect and undo the shift; with registration OFF the misalignment causes spurious 'added' candidate rejections (many small slivers from the offset) and drives polygon vertex counts up due to less-clean masks.

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shifted (2, 3): prod-equiv | 131 | 1 | 2 | 2 | 136 | 9 | 16.3 | 0.76 | 0 | 2.0 | 3.0 | False |
| shifted (2, 3): new-default | 131 | 1 | 2 | 2 | 136 | 9 | 15.2 | 0.76 | 0 | 2.0 | 3.0 | True |
| shifted (4, 0): prod-equiv | 130 | 2 | 2 | 2 | 136 | 9 | 15.7 | 0.77 | 0 | 4.0 | 0.0 | False |
| shifted (4, 0): new-default | 130 | 2 | 2 | 2 | 136 | 9 | 15.2 | 0.77 | 0 | 4.0 | 0.0 | True |
| shifted (0, -5): prod-equiv | 130 | 2 | 2 | 2 | 136 | 9 | 15.5 | 0.74 | 0 | -1.0 | 1.0 | False |
| shifted (0, -5): new-default | 130 | 2 | 2 | 2 | 136 | 9 | 15.4 | 0.74 | 0 | -1.0 | 1.0 | False |

## Per-config detail

### production-equiv (all OFF)

- counts: {'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2}
- registration: {'dy': 0.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.9578, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 134
- vertex stats: n=136 min=4 med=9 mean=15.3 max=161
- confidence stats: n=136 min=0.62 med=0.84 mean=0.82 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### new-default (all ON)

- counts: {'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2}
- registration: {'dy': 0.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.9578, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 134
- vertex stats: n=136 min=4 med=9 mean=15.2 max=147
- confidence stats: n=136 min=0.62 med=0.84 mean=0.82 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### ablation: registration OFF

- counts: {'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2}
- registration: {'dy': 0.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.9578, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 134
- vertex stats: n=136 min=4 med=9 mean=15.2 max=147
- confidence stats: n=136 min=0.62 med=0.84 mean=0.82 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### ablation: smoothing OFF

- counts: {'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2}
- registration: {'dy': 0.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.9578, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 134
- vertex stats: n=136 min=4 med=9 mean=15.3 max=161
- confidence stats: n=136 min=0.62 med=0.84 mean=0.82 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### ablation: surface delta-E OFF

- counts: {'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2}
- registration: {'dy': 0.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.9578, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 134
- vertex stats: n=136 min=4 med=9 mean=15.2 max=147
- confidence stats: n=136 min=0.62 med=0.84 mean=0.82 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (2, 3): prod-equiv

- counts: {'unchanged': 131, 'modified': 1, 'demolished': 2, 'added': 2}
- registration: {'dy': 2.0, 'dx': 3.0, 'confidence': 1.0, 'iou_before': 0.8281, 'iou_after': 0.9576, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 131
- vertex stats: n=136 min=4 med=9 mean=16.3 max=267
- confidence stats: n=136 min=0.60 med=0.76 mean=0.75 max=1.00
- added_rejected: {'too_small': 204, 'baseline_overlap': 7, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (2, 3): new-default

- counts: {'unchanged': 131, 'modified': 1, 'demolished': 2, 'added': 2}
- registration: {'dy': 2.0, 'dx': 3.0, 'confidence': 1.0, 'iou_before': 0.8281, 'iou_after': 0.9576, 'applied': True}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 131
- vertex stats: n=136 min=4 med=9 mean=15.2 max=147
- confidence stats: n=136 min=0.60 med=0.76 mean=0.75 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (4, 0): prod-equiv

- counts: {'unchanged': 130, 'modified': 2, 'demolished': 2, 'added': 2}
- registration: {'dy': 4.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.8477, 'iou_after': 0.9578, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 130
- vertex stats: n=136 min=4 med=9 mean=15.7 max=193
- confidence stats: n=136 min=0.58 med=0.77 mean=0.76 max=1.00
- added_rejected: {'too_small': 794, 'baseline_overlap': 4, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (4, 0): new-default

- counts: {'unchanged': 130, 'modified': 2, 'demolished': 2, 'added': 2}
- registration: {'dy': 4.0, 'dx': 0.0, 'confidence': 1.0, 'iou_before': 0.8477, 'iou_after': 0.9578, 'applied': True}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 130
- vertex stats: n=136 min=4 med=9 mean=15.2 max=147
- confidence stats: n=136 min=0.58 med=0.77 mean=0.76 max=1.00
- added_rejected: {'too_small': 0, 'baseline_overlap': 0, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (0, -5): prod-equiv

- counts: {'unchanged': 130, 'modified': 2, 'demolished': 2, 'added': 2}
- registration: {'dy': -1.0, 'dx': 1.0, 'confidence': 0.293, 'iou_before': 0.7994, 'iou_after': 0.776, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 130
- vertex stats: n=136 min=4 med=9 mean=15.5 max=175
- confidence stats: n=136 min=0.59 med=0.74 mean=0.74 max=1.00
- added_rejected: {'too_small': 551, 'baseline_overlap': 10, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

### shifted (0, -5): new-default

- counts: {'unchanged': 130, 'modified': 2, 'demolished': 2, 'added': 2}
- registration: {'dy': -1.0, 'dx': 1.0, 'confidence': 0.293, 'iou_before': 0.7994, 'iou_after': 0.776, 'applied': False}
- surface_change_available: True
- surface_changed (in unchanged): 0
- surface_delta_e samples: 130
- vertex stats: n=136 min=4 med=9 mean=15.4 max=159
- confidence stats: n=136 min=0.59 med=0.74 mean=0.74 max=1.00
- added_rejected: {'too_small': 551, 'baseline_overlap': 10, 'centroid_near_baseline': 0, 'too_short': 0, 'no_lidar_coverage_emitted_as_candidate': 0}

