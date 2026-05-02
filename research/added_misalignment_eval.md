# Research: 'added' false-positive fix — all 5 demos

Evaluates the wider near-baseline gate (DILATE_PX 8 → 24) plus the new majority-inside-dilation guard. Compares pre-fix vs the new defaults on every cached production demo. Goal: drop false-positive 'added' counts on dense demos (Brooklyn, Hudson Yards, East Village) without dropping legitimate new-construction additions elsewhere.

## Summary

| demo | before u/m/d/a | after u/m/d/a | Δ added | centroid_near | majority_inside |
| --- | --- | --- | --- | --- | --- |
| 100 E 21st St Brooklyn, NY 11226 (`5f079d78`) | 134/0/0/2 | 134/0/0/0 | 2 → 0 | 0 | 2 |
| 15 Hudson Yards, New York, NY 10 (`6b3e42cd`) | 33/6/0/2 | 33/6/0/0 | 2 → 0 | 2 | 0 |
| 5-49 Borden Ave, Long Island Cit (`c0d396fe`) | 113/1/0/1 | 113/1/0/0 | 1 → 0 | 1 | 0 |

## Skipped demos

- `36e628d469fc43c1999af98b8568470c` (20 Cooper Square, New York, NY 10003): no cached ortho covers demo centroid
- `5885e4d33f1342fc9ef58cf0fae84733` (240 Bedford Ave, Brooklyn, NY 11211): no cached ortho covers demo centroid

