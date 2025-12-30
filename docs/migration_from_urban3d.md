# Migration from Urban3D-DeepRecon

This package is intended to extract reusable, non-UI logic from `../Urban3D-DeepRecon`.

## Vendored modules

Currently vendored under `src/citylens_core/vendor/urban3d/`:

- `config.py`
- `unet.py`
- `segmentation.py` (adapted, lazy-imports)

These are copied to preserve compatibility with the original U-Net implementation, but are not
imported by default to keep base installs lightweight.

## Excluded

- Streamlit dashboard UI (`dashboard.py`)

## Assets

SAM2 assets are downloaded on demand and are **not committed**.
Use `make sam2-assets` to download the default small model assets.
