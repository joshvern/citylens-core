# Research: change-stage improvements — all 5 demos

Extends `change_algo_improvements_eval.md` (Brooklyn-only) to every demo in `citylens-engine/deploy/demo_runs.json` that has a cached orthophoto in `gs://citylens-001-artifacts/inputs/`. Demos without cached coverage are listed with the reason — re-fetching from NYS WMS is out of scope here.

## Summary table — covered demos

| demo | feat | u/m/d/a | shift(2,3) too_small off→on | vert_max off→on | surface_avail |
| --- | --- | --- | --- | --- | --- |
| 100 E 21st St Brooklyn, NY 11226 (`5f079d78`) | 136 | 134/0/0/2 | 204 → 0 | 161 → 147 | yes |
| 15 Hudson Yards, New York, NY 10001 (`6b3e42cd`) | 41 | 31/8/0/2 | 673 → 0 | 208 → 164 | yes |
| 5-49 Borden Ave, Long Island City,  (`c0d396fe`) | 115 | 113/1/0/1 | 239 → 1 | 209 → 183 | yes |

## Skipped demos

- `36e628d4` 20 Cooper Square, New York, NY 10003 — no cached ortho covers demo centroid; would need NYS WMS fetch
- `5885e4d3` 240 Bedford Ave, Brooklyn, NY 11211 — no cached ortho covers demo centroid; would need NYS WMS fetch

## Per-demo detail

### 100 E 21st St Brooklyn, NY 11226 (`5f079d78d89c`)

- ortho: `/tmp/citylens-eval/inputs/856bd45e3b6d9ffac14fccabfcd34aabaf5381b17231470993da4fd214cd2aa3.tif`, cropped shape `[1024, 703]`
- imagery mask coverage: 345086 px
- baseline raster coverage: 330536 px
- baseline features (with `source_gdb`): 134

**Aligned configs:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production-equiv (all OFF) | 134 | 0 | 0 | 2 | 136 | 9 | 15.3 | 0.84 | 0 | 0.0 | 0.0 | False |
| new-default (all ON) | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: registration OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: smoothing OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.3 | 0.84 | 0 | 0.0 | 0.0 | False |
| ablation: surface delta-E OFF | 134 | 0 | 0 | 2 | 136 | 9 | 15.2 | 0.84 | 0 | 0.0 | 0.0 | False |

**Shift tests:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shifted (2, 3): prod-equiv | 131 | 1 | 2 | 2 | 136 | 9 | 16.3 | 0.76 | 0 | 2.0 | 3.0 | False |
| shifted (2, 3): new-default | 131 | 1 | 2 | 2 | 136 | 9 | 15.2 | 0.76 | 0 | 2.0 | 3.0 | True |
| shifted (4, 0): prod-equiv | 130 | 2 | 2 | 2 | 136 | 9 | 15.7 | 0.77 | 0 | 4.0 | 0.0 | False |
| shifted (4, 0): new-default | 130 | 2 | 2 | 2 | 136 | 9 | 15.2 | 0.77 | 0 | 4.0 | 0.0 | True |

- `production-equiv (all OFF)`: vert max=161 med=9 | added_rejected.too_small=0 | reg_applied=False
- `new-default (all ON)`: vert max=147 med=9 | added_rejected.too_small=0 | reg_applied=False
- `shifted (2, 3): prod-equiv`: vert max=267 med=9 | added_rejected.too_small=204 | reg_applied=False
- `shifted (2, 3): new-default`: vert max=147 med=9 | added_rejected.too_small=0 | reg_applied=True
- `shifted (4, 0): prod-equiv`: vert max=193 med=9 | added_rejected.too_small=794 | reg_applied=False
- `shifted (4, 0): new-default`: vert max=147 med=9 | added_rejected.too_small=0 | reg_applied=True

### 15 Hudson Yards, New York, NY 10001 (`6b3e42cd42d2`)

- ortho: `/tmp/citylens-eval/inputs/a7a268c4a92361ce173e1f752e70d75bf856e35b5396b0cd056536e92db6585b.tif`, cropped shape `[1024, 1024]`
- imagery mask coverage: 331764 px
- baseline raster coverage: 328311 px
- baseline features (with `source_gdb`): 39

**Aligned configs:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production-equiv (all OFF) | 31 | 8 | 0 | 2 | 41 | 11 | 19.8 | 0.81 | 0 | 0.0 | 0.0 | False |
| new-default (all ON) | 31 | 8 | 0 | 2 | 41 | 11 | 17.8 | 0.81 | 0 | 0.0 | 0.0 | False |
| ablation: registration OFF | 31 | 8 | 0 | 2 | 41 | 11 | 17.8 | 0.81 | 0 | 0.0 | 0.0 | False |
| ablation: smoothing OFF | 31 | 8 | 0 | 2 | 41 | 11 | 19.8 | 0.81 | 0 | 0.0 | 0.0 | False |
| ablation: surface delta-E OFF | 31 | 8 | 0 | 2 | 41 | 11 | 17.8 | 0.81 | 0 | 0.0 | 0.0 | False |

**Shift tests:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shifted (2, 3): prod-equiv | 30 | 9 | 0 | 2 | 41 | 11 | 19.8 | 0.76 | 0 | 2.0 | 3.0 | False |
| shifted (2, 3): new-default | 30 | 9 | 0 | 2 | 41 | 11 | 17.7 | 0.76 | 0 | 2.0 | 3.0 | True |
| shifted (4, 0): prod-equiv | 30 | 9 | 0 | 1 | 40 | 11 | 16.6 | 0.79 | 0 | 4.0 | 0.0 | False |
| shifted (4, 0): new-default | 30 | 9 | 0 | 2 | 41 | 11 | 17.7 | 0.79 | 0 | 4.0 | 0.0 | True |

- `production-equiv (all OFF)`: vert max=208 med=11 | added_rejected.too_small=0 | reg_applied=False
- `new-default (all ON)`: vert max=164 med=11 | added_rejected.too_small=0 | reg_applied=False
- `shifted (2, 3): prod-equiv`: vert max=200 med=11 | added_rejected.too_small=673 | reg_applied=False
- `shifted (2, 3): new-default`: vert max=162 med=11 | added_rejected.too_small=0 | reg_applied=True
- `shifted (4, 0): prod-equiv`: vert max=214 med=11 | added_rejected.too_small=50 | reg_applied=False
- `shifted (4, 0): new-default`: vert max=158 med=11 | added_rejected.too_small=0 | reg_applied=True

### 5-49 Borden Ave, Long Island City, NY 11101 (`c0d396fe7478`)

- ortho: `/tmp/citylens-eval/inputs/54a92192c760339fff622782228a541018141d7dd50e9f26728657ae5dd5e4b9.tif`, cropped shape `[952, 1024]`
- imagery mask coverage: 303891 px
- baseline raster coverage: 301887 px
- baseline features (with `source_gdb`): 114

**Aligned configs:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| production-equiv (all OFF) | 113 | 1 | 0 | 1 | 115 | 8 | 11.8 | 0.76 | 0 | 0.0 | 0.0 | False |
| new-default (all ON) | 113 | 1 | 0 | 1 | 115 | 8 | 11.6 | 0.76 | 0 | 0.0 | 0.0 | False |
| ablation: registration OFF | 113 | 1 | 0 | 1 | 115 | 8 | 11.6 | 0.76 | 0 | 0.0 | 0.0 | False |
| ablation: smoothing OFF | 113 | 1 | 0 | 1 | 115 | 8 | 11.8 | 0.76 | 0 | 0.0 | 0.0 | False |
| ablation: surface delta-E OFF | 113 | 1 | 0 | 1 | 115 | 8 | 11.6 | 0.76 | 0 | 0.0 | 0.0 | False |

**Shift tests:**

| config | unchanged | modified | demolished | added | feat_total | vert_med | vert_mean | conf_med | surf_changed | reg_dy | reg_dx | reg_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shifted (2, 3): prod-equiv | 107 | 5 | 2 | 1 | 115 | 8 | 11.9 | 0.70 | 0 | 2.0 | 3.0 | False |
| shifted (2, 3): new-default | 107 | 5 | 2 | 1 | 115 | 8 | 11.6 | 0.70 | 0 | 2.0 | 3.0 | True |
| shifted (4, 0): prod-equiv | 103 | 8 | 3 | 1 | 115 | 8 | 11.7 | 0.67 | 0 | 4.0 | 0.0 | False |
| shifted (4, 0): new-default | 103 | 8 | 3 | 1 | 115 | 8 | 11.5 | 0.67 | 0 | 4.0 | 0.0 | True |

- `production-equiv (all OFF)`: vert max=209 med=8 | added_rejected.too_small=1 | reg_applied=False
- `new-default (all ON)`: vert max=183 med=8 | added_rejected.too_small=1 | reg_applied=False
- `shifted (2, 3): prod-equiv`: vert max=226 med=8 | added_rejected.too_small=239 | reg_applied=False
- `shifted (2, 3): new-default`: vert max=181 med=8 | added_rejected.too_small=1 | reg_applied=True
- `shifted (4, 0): prod-equiv`: vert max=198 med=8 | added_rejected.too_small=606 | reg_applied=False
- `shifted (4, 0): new-default`: vert max=175 med=8 | added_rejected.too_small=1 | reg_applied=True

