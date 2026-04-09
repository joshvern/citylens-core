# citylens-core

Installable, reusable pipeline core extracted from `Urban3D-DeepRecon`.

`citylens-core` is an independently runnable repo under the shared
`/home/josh/citylens` workspace. It uses its own repo-local `.venv` and is meant
to be opened directly in VS Code, or through a multi-root workspace that keeps
`citylens-core`, `citylens-engine`, and `citylens-web` as separate roots.

Active product development happens across `citylens-core`, `citylens-engine`,
and `citylens-web`. `Urban3D-DeepRecon` is kept as a reference implementation
for algorithms and legacy workflow comparison.

This repo is the reusable pipeline library, not the deployment target. The
engine repo owns the API and worker deployment surfaces that consume this core.

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
