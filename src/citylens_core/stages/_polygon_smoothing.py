"""Douglas-Peucker simplification for change-detection polygons.

`rasterio.features.shapes()` produces saw-tooth polygon coordinates
that follow pixel boundaries exactly. This module simplifies them via
shapely's `Polygon.simplify(preserve_topology=True)` so the output
GeoJSON looks like clean buildings instead of pixelated rasters.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon


def _is_ring_malformed(ring: list[list[float]]) -> bool:
    """A GeoJSON ring needs at least 4 coordinates (3 unique + closing)."""
    if ring is None:
        return True
    if len(ring) < 4:
        return True
    return False


def _close_ring(ring: list[list[float]]) -> list[list[float]]:
    """Ensure first coord == last coord (GeoJSON closure)."""
    if not ring:
        return ring
    first = ring[0]
    last = ring[-1]
    if len(first) < 2 or len(last) < 2:
        return ring
    if first[0] != last[0] or first[1] != last[1]:
        return ring + [list(first)]
    return ring


def _coords_to_list(coords) -> list[list[float]]:
    """Turn a shapely CoordinateSequence into a plain list of [x, y] lists."""
    out: list[list[float]] = []
    for pt in coords:
        if pt is None:
            continue
        if len(pt) < 2:
            continue
        out.append([float(pt[0]), float(pt[1])])
    return out


def simplify_polygon_coords(
    rings: list[list[list[float]]],
    *,
    tolerance: float,
) -> list[list[list[float]]]:
    """Run Douglas-Peucker simplification on one polygon's coordinate
    rings (outer ring + holes). Always returns valid, closed rings.
    Falls back to the input if anything goes wrong."""
    if not rings:
        return []

    for ring in rings:
        if _is_ring_malformed(ring):
            return rings

    if tolerance is None or tolerance <= 0:
        return [_close_ring([list(pt) for pt in ring]) for ring in rings]

    outer = rings[0]
    holes = rings[1:] if len(rings) > 1 else []

    try:
        polygon = Polygon(outer, holes)
    except Exception:
        return rings

    try:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
    except Exception:
        return rings

    if simplified is None:
        return rings
    if simplified.is_empty:
        return rings
    if simplified.geom_type != "Polygon":
        return rings

    exterior = simplified.exterior
    if exterior is None:
        return rings

    new_outer = _coords_to_list(exterior.coords)
    if len(new_outer) < 4:
        return rings

    new_outer = _close_ring(new_outer)

    new_holes: list[list[list[float]]] = []
    for interior in simplified.interiors:
        hole_coords = _coords_to_list(interior.coords)
        if len(hole_coords) < 4:
            continue
        new_holes.append(_close_ring(hole_coords))

    return [new_outer] + new_holes


def estimate_pixel_tolerance_in_world_units(
    transform,
    *,
    pixel_tolerance: float = 0.5,
) -> float:
    """Convert a pixel-space tolerance into world units using the
    rasterio Affine transform's pixel size."""
    if transform is None:
        return float(pixel_tolerance)

    try:
        a = float(transform.a)
        b = float(transform.b)
        e = float(transform.e)
    except AttributeError:
        try:
            a = float(transform[0])
            b = float(transform[1])
            e = float(transform[4])
        except Exception:
            return float(pixel_tolerance)
    except Exception:
        return float(pixel_tolerance)

    if (
        abs(b) < 1e-12
        and math.isclose(abs(a), 1.0, abs_tol=1e-9)
        and math.isclose(abs(e), 1.0, abs_tol=1e-9)
    ):
        return float(pixel_tolerance)

    pixel_size = math.sqrt(a * a + b * b)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        return float(pixel_tolerance)

    return float(pixel_tolerance) * pixel_size
