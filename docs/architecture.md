# Architecture

`citylens-core` is the reusable pipeline library. It is not the deployment
surface; the engine repo owns the API and worker entrypoints that consume this
package.

The pipeline is implemented as a sequence of stages:

1. `resolve` – validate request and set up `work_dir`
2. `fetch` – resolve local inputs or download explicit URLs into `work_dir`; record raster metadata when available
3. `segment` – run SAM2 on the orthophoto and optional baseline mask
4. `refine` – normalize masks, clean morphology, and optionally rasterize `baseline_footprints.geojson` guidance
5. `change` – produce `change.geojson` in pixel space or georeferenced coordinates, depending on input metadata
6. `reconstruct` – produce `mesh.ply`; use `work_dir/lidar.las` as the primary geometry source and fall back to mask heights
7. `render` – produce `preview.png`

Failures are recorded into `run_summary.json`. The pipeline does not write placeholder artifacts.
Successful summaries also carry `qa` and `performance` sections for downstream parity tracking.
