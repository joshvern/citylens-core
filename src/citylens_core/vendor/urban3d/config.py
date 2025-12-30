"""Vendored from ../Urban3D-DeepRecon/src/config.py.

Only included to preserve compatibility with the original codebase.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "lidar": "https://example.ny.gov/gis/lidar/YOUR_COUNTY/YEAR/sample_tile.las",
    "satellite": "https://example.ny.gov/gis/ortho/YOUR_COUNTY/YEAR/sample_ortho.tif",
}

FILENAMES = {
    "lidar": os.path.join(DATA_DIR, "sample_lidar.las"),
    "satellite": os.path.join(DATA_DIR, "sample_satellite.tif"),
}

OUTPUT_MESH = os.path.join(BASE_DIR, "output_mesh.ply")
