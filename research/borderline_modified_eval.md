# Research: borderline-modified reclassification — all 5 demos

Evaluates the new `_modified_borderline_margin` two-stage classifier added to `stage_change`. Compares the pre-fix behavior (margin=0) against the new default (margin=0.05) on every cached production demo in `gs://citylens-001-artifacts/runs/`.

**Hypothesis**: most production "modified" features have IoU clustered just below the (adaptive) unchanged threshold; demoting that cluster to unchanged should drop the per-tile modified rate from 10–25 % toward a more plausible single-digit baseline without losing real change signal (features deeper into the modified band stay modified).

## Before / after

| demo | before u/m/d/a | after u/m/d/a | Δ modified | kept_by_surface | unchanged_iou_used | border_band |
| --- | --- | --- | --- | --- | --- | --- |
| 100 E 21st St Brooklyn, NY 11226 (`5f079d78`) | 134/0/0/2 | 134/0/0/2 | 0 → 0 | 0 | 0.4 | [0.350, 0.4) |
| 15 Hudson Yards, New York, NY 10 (`6b3e42cd`) | 31/8/0/2 | 33/6/0/2 | 8 → 6 | 0 | 0.4 | [0.350, 0.4) |
| 5-49 Borden Ave, Long Island Cit (`c0d396fe`) | 113/1/0/1 | 113/1/0/1 | 1 → 1 | 0 | 0.4 | [0.350, 0.4) |

## Skipped demos

- `36e628d4` 20 Cooper Square, New York, NY 10003 — no cached ortho covers demo centroid
- `5885e4d3` 240 Bedford Ave, Brooklyn, NY 11211 — no cached ortho covers demo centroid

## Per-demo detail

### 100 E 21st St Brooklyn, NY 11226 (`5f079d78d89c`)

- `pre-fix (margin=0)`: counts={'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2} unchanged_iou_used=0.4 borderline={'to_unchanged': 0, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}
- `new-default (margin=0.05)`: counts={'unchanged': 134, 'modified': 0, 'demolished': 0, 'added': 2} unchanged_iou_used=0.4 borderline={'to_unchanged': 0, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}

### 15 Hudson Yards, New York, NY 10001 (`6b3e42cd42d2`)

- `pre-fix (margin=0)`: counts={'unchanged': 31, 'modified': 8, 'demolished': 0, 'added': 2} unchanged_iou_used=0.4 borderline={'to_unchanged': 0, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}
- `new-default (margin=0.05)`: counts={'unchanged': 33, 'modified': 6, 'demolished': 0, 'added': 2} unchanged_iou_used=0.4 borderline={'to_unchanged': 2, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}

### 5-49 Borden Ave, Long Island City, NY 11101 (`c0d396fe7478`)

- `pre-fix (margin=0)`: counts={'unchanged': 113, 'modified': 1, 'demolished': 0, 'added': 1} unchanged_iou_used=0.4 borderline={'to_unchanged': 0, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}
- `new-default (margin=0.05)`: counts={'unchanged': 113, 'modified': 1, 'demolished': 0, 'added': 1} unchanged_iou_used=0.4 borderline={'to_unchanged': 0, 'kept_by_surface_change': 0, 'kept_lenient_no_surface_signal': 0}

