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
- When baseline footprints select prompted SAM2 and `change` is requested,
  CityLens also runs automatic SAM2 on the current orthophoto for
  added-building discovery. The masks stay separate: prompted output drives
  existing-footprint IoU classifications, while automatic output is consumed
  only by the `added` path. `CITYLENS_SAM2_ADDED_DISCOVERY=false` restores the
  legacy prompted-only behavior for explicit ablations; discovery is otherwise
  required and a discovery failure fails the change run. Edge-connected
  automatic components are rejected by default to suppress tile-scale
  road/background masks (`CITYLENS_CHANGE_ADDED_REJECT_BORDER_TOUCHING=false`
  disables that gate).
