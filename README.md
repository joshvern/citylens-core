# citylens-core

Installable, reusable pipeline core extracted from `Urban3D-DeepRecon`.

**Goals**

- No GCP code; everything writes artifacts locally under a `work_dir`.
- Segmentation backends: `unet`, `smp`, `sam2` (v2.1).
- Always produces standard artifacts in `work_dir`:
  - `preview.png`
  - `change.geojson`
  - `mesh.ply` (placeholder if mesh deps missing)
  - `run_summary.json`

## Install

Quickstart (no `requirements.txt`; everything is driven by `pyproject.toml`):

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sam2]"
```

Base (dev):

```bash
pip install -e ".[dev]"
```

With SAM2:

```bash
pip install -e ".[dev,sam2]"
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

- If `sam2` is not installed or SAM2 assets are missing, the pipeline will emit placeholders and
  record warnings in `run_summary.json` (it should not crash).
