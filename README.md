# citylens-core

The **reusable Python pipeline library** behind CityLens — segmentation,
change detection, and 3D mesh reconstruction from aerial imagery. Used in
production by [`citylens-engine`](https://github.com/joshvern/citylens-engine)
to power the live product at **https://www.citylens.dev**.

`citylens-core` is the algorithm layer: pure Python, no GCP, no API. It
takes a `CitylensRequest` plus a `work_dir` and returns standard artifacts
on disk. The engine repo owns the deployment surface (API, worker, auth,
quotas, GCS); this repo is meant to be `pip install`-able and embedded into
any caller — server, batch job, or notebook.

Companion repos:

- [`citylens-engine`](https://github.com/joshvern/citylens-engine) — API,
  worker, auth, quotas, artifact storage. Wraps this library.
- [`citylens-web`](https://github.com/joshvern/citylens-web) — Next.js
  product frontend at https://www.citylens.dev.

`Urban3D-DeepRecon` is kept as a read-only reference implementation for
algorithms and legacy workflow comparison.

**Goals**

- No GCP code; everything writes artifacts locally under a `work_dir`.
- Segmentation backend: `sam2`.
- Always produces standard artifacts in `work_dir`:
  - `preview.png`
  - `change.geojson`
  - `mesh.ply`
  - `run_summary.json`

## Install

Quickstart (no `requirements.txt`; everything is driven by `pyproject.toml`):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sam2]"
```

When working from the shared `/home/josh/citylens` parent folder, keep this
repo's environment isolated:

```bash
cd /home/josh/citylens/citylens-core
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sam2]"
```

VS Code should open `citylens-core` directly, or use a workspace that contains
`citylens-core`, `citylens-engine`, and `citylens-web` as separate folders. Do
not rely on the parent folder to infer the interpreter or package root.

Base (dev):

```bash
pip install -e ".[dev]"
```

With SAM2:

```bash
pip install -e ".[dev,sam2]"
```

With optional LiDAR helpers:

```bash
pip install -e ".[dev,sam2,lidar]"
```

## Download SAM2 assets

```bash
make sam2-assets
```

Alternatively:

```bash
make install-sam2
make sam2-assets
```

## Example

```python
from pathlib import Path

from citylens_core import CitylensRequest, run_citylens

work_dir = Path("/tmp/citylens-run")
req = CitylensRequest(address="350 5th Ave, New York, NY", segmentation_backend="sam2")

artifacts = run_citylens(req, work_dir)
print(artifacts)
```

## Notes

- If `sam2` is not installed or SAM2 assets are missing, the pipeline fails and writes only
  `run_summary.json`.
- `change.geojson` is georeferenced only when the fetched/provided orthophoto includes raster
  CRS + transform metadata; otherwise the output stays explicit pixel space with
  `properties.crs = "pixel"`.
- `mesh.ply` uses `work_dir/lidar.las` when available and `laspy` is installed; otherwise it
  falls back to a deterministic mask-height mesh.
- `run_summary.json` includes `qa` and `performance` sections with optional parity metrics.
- `work_dir/current_footprints.geojson` is an optional, purely local semantic
  current-building source. Its FeatureCollection geometries must already be in
  the orthophoto CRS; `construction_year`, `last_status_type`, `geom_source`,
  `base_bbl`, `mappluto_bbl`, and `source_dataset` properties are forwarded as
  provenance. When usable, its rasterized union is preferred for current
  building presence and dated post-baseline features directly produce
  `added`/`modified` events subject to the same 60 m² commercial noise floor
  as generic discovery. A source MultiPolygon remains one event with its total
  rasterized area. Edge-clipped semantic changes are omitted as incomplete;
  edge-clipped baseline `modified`/`demolished` calls are likewise omitted,
  while confirmed unchanged edge presence remains visible. A valid empty
  collection is authoritative.
- Without semantic current footprints, prompted `change` runs also run
  automatic SAM2 on the current orthophoto for added-building discovery. The
  masks stay separate: prompted output drives existing-footprint IoU while the
  automatic output is consumed only by the `added` path.
  `CITYLENS_SAM2_ADDED_DISCOVERY=false` restores prompted-only behavior for
  explicit ablations; discovery otherwise fails honestly with the run.
  Edge-connected automatic components are rejected by default to suppress
  tile-scale road/background masks
  (`CITYLENS_CHANGE_ADDED_REJECT_BORDER_TOUCHING=false` disables that gate).
- **LiDAR epoch semantics (v0.3.25):** production LiDAR is baseline-epoch
  (2017), so a *flat* LiDAR reading where SAM2 sees a current building is
  positive evidence of new construction — such `added` events get boosted
  confidence (`baseline_lidar_flat: true`), while a tall-in-baseline reading
  demotes to `candidate_added`. `CITYLENS_CHANGE_ADDED_MAX_BASELINE_HEIGHT_M`
  (default 2.0; legacy `CITYLENS_CHANGE_ADDED_MIN_HEIGHT_M` honored as
  fallback) sets the flat threshold. The demolished-rescue check is epoch-gated
  **off** by default (`CITYLENS_CHANGE_DEMOLISHED_RESCUE_LIDAR_EPOCH=baseline`;
  set `current` only if your LAS is current-epoch). `height_m` on features is
  baseline-epoch height.
- Other change-stage tuning: `CITYLENS_CHANGE_ADDED_EXG_THRESHOLD` (median
  ExG vegetation reject on the added path, default 30),
  `CITYLENS_CHANGE_MIN_AREA_M2` (default 60; areas are true ground m² — the
  Web-Mercator cos²(lat) correction is applied), per-footprint local
  registration (±2 px windowed slide; `local_shift_px` property,
  `registration.saturated` QA flag), `CITYLENS_SAM2_PROMPT_BATCH_SIZE`
  (default 32).
- `run_summary.json` QA reports `mask_xor_f1` (agreement between the change
  classification and the raw mask XOR — a consistency signal, **not** an
  accuracy metric; its reference is circular). The old `change_polygon_f1`
  name is kept as a deprecated alias. For an external accuracy check, use the
  DOB weak-label harness in `citylens-parcel-intel/scripts/weak_label_eval.py`.
