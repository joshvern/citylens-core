# Migration from Urban3D-DeepRecon

This package is intended to extract reusable, non-UI logic from `../Urban3D-DeepRecon`.
The monolith is now treated as a reference implementation, not the active product surface.

`citylens-core` is a reusable library repo, not a deployment target. It runs
independently with its own repo-local `.venv` and is consumed by the engine and
web repos under the shared `/home/josh/citylens` workspace.

When using VS Code, open `citylens-core` as its own folder or as one root in a
multi-root workspace. Do not depend on the parent `/home/josh/citylens` folder
to infer the active interpreter or package root for this repo.

## Vendored modules

Currently vendored under `src/citylens_core/vendor/urban3d/`:

- `config.py`
- `unet.py`
- `segmentation.py` (adapted, lazy-imports)

These are copied to preserve compatibility with the original U-Net implementation, but are not
imported by default to keep base installs lightweight.
The active pipeline uses SAM2 plus deterministic core-side refinement instead.

## Fixed Reference Case

The modular stack uses a fixed acceptance case for parity work:

- `100 E 21st St Brooklyn, NY 11226`

For core, the relevant acceptance signals are surfaced through `run_summary.json`:

- `qa.mask_iou`
- `qa.change_polygon_f1`
- `qa.mesh_footprint_iou`
- `qa.parity_status`

These metrics are consumed by the engine parity harness and surfaced by the web UI.

## Excluded

- Streamlit dashboard UI (`dashboard.py`)

## Workspace Organization

The active product work is split across:

- `citylens-core` for the reusable pipeline library
- `citylens-engine` for the API and worker deployment surfaces
- `citylens-web` for the browser product surface

Keep their environments and editor contexts separate even when they live under
the same `/home/josh/citylens` parent directory.

## Assets

SAM2 assets are downloaded on demand and are **not committed**.
Use `make sam2-assets` to download the default small model assets.
For release deployments, set `CITYLENS_ASSETS_ROOT` to a stable runtime path instead of relying on the current working directory.
These SAM2 assets are required for worker/precompute generation, but they are **not** required to serve baked demo bundles from `citylens-engine/deploy/demo_artifacts`.
