# Architecture

The pipeline is implemented as a sequence of stages:

1. `resolve` – validate request and set up `work_dir`
2. `fetch` – acquire inputs (currently stubbed; writes placeholder imagery)
3. `segment` – produce a binary structure/building mask
4. `change` – produce `change.geojson` (stubbed)
5. `reconstruct` – produce `mesh.ply` (placeholder if `open3d` not installed)
6. `render` – produce `preview.png`

All stages are best-effort: failures are recorded into `run_summary.json` and placeholders are written.
