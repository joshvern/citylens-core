# Architecture

`citylens-core` is the reusable pipeline library. It is not the deployment
surface; the engine repo owns the API and worker entrypoints that consume this
package.

The pipeline is implemented as a sequence of stages:

1. `resolve` – validate request and set up `work_dir`
2. `fetch` – resolve local inputs or download explicit URLs into `work_dir`; record raster metadata when available
3. `segment` – run SAM2 on the orthophoto and optional baseline mask. Prompted
   change runs keep two current-image masks: a baseline-prompted classification
   mask and a separate automatic added-building discovery mask. The paired path
   shares one SAM2 model load. A usable staged `current_footprints.geojson`
   replaces the automatic discovery pass while SAM still runs for imagery QA.
4. `refine` – independently normalize the classification/discovery masks,
   clean morphology, and optionally rasterize `baseline_footprints.geojson`
   plus the already-ortho-CRS `current_footprints.geojson`
5. `change` – prefer the semantic current-footprint union for baseline presence;
   emit dated post-baseline current features as source-aware `added` or
   `modified` events; otherwise fall back to SAM discovery; produce
   `change.geojson` in pixel space or georeferenced coordinates
6. `reconstruct` – produce `mesh.ply`; use `work_dir/lidar.las` as the primary geometry source and fall back to mask heights
7. `render` – produce `preview.png`

Failures are recorded into `run_summary.json`. The pipeline does not write placeholder artifacts.
Successful summaries also carry `qa` and `performance` sections for downstream parity tracking.
Discovery QA includes its source/status plus raw and refined coverage,
component count, and largest-component diagnostics.
