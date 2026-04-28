from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary
from ..io.lidar import build_height_map_from_lidar


# ----------------------------------------------------------------------
# LOD1 per-building extrusions
# ----------------------------------------------------------------------
#
# When we have both per-building classification (change.geojson) and a
# pixel-aligned LiDAR height grid (populated by stage_refine), we emit
# a true per-footprint extrusion mesh: walls + fan-triangulated roof cap
# per building. Each building gets its own roof height from the 95th
# percentile of LiDAR returns inside its rasterized footprint. Buildings
# classified "demolished" are dropped so the mesh matches the 2024
# reality, not the 2017 baseline.
#
# This replaces the old "downsampled heightfield of the whole ortho"
# behavior which produced a single flat blob with no building
# separation. The heightfield path is kept as a fallback for test fixtures
# and paths where the LiDAR sample is unavailable.


_LOD1_DEFAULT_HEIGHT_M = 8.0  # ~3 stories — used when a polygon has no LiDAR coverage
_LOD1_ROOF_PERCENTILE = 95.0

# Per-vertex RGB so a 3D viewer can color buildings by change classification
# without needing a separate colormap. Same palette as the preview PNG legend
# in stages/render.py — keep them in sync. demolished is skipped from the
# mesh entirely so it doesn't need a color here.
_LOD1_BUILDING_COLOR: dict[str, tuple[int, int, int]] = {
    "unchanged": (140, 140, 140),
    "modified": (255, 200, 0),
    "added": (0, 200, 60),
}
_LOD1_DEFAULT_COLOR: tuple[int, int, int] = (140, 140, 140)


def _load_change_features(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    feats = payload.get("features")
    if not isinstance(feats, list):
        return []
    return [f for f in feats if isinstance(f, dict) and f.get("geometry")]


def _feature_rings_pixel(
    geom: dict[str, Any], inv_transform: Any
) -> list[list[tuple[float, float]]]:
    """Return exterior rings of a GeoJSON Polygon/MultiPolygon in pixel
    coordinates. Holes are dropped for LOD1 MVP.
    """
    typ = geom.get("type")
    coords = geom.get("coordinates") or []
    crs_rings: list[list[list[float]]] = []
    if typ == "Polygon":
        if coords:
            crs_rings.append(coords[0])
    elif typ == "MultiPolygon":
        for poly in coords:
            if poly:
                crs_rings.append(poly[0])
    else:
        return []

    out: list[list[tuple[float, float]]] = []
    for ring in crs_rings:
        px: list[tuple[float, float]] = []
        for pt in ring:
            if not pt or len(pt) < 2:
                continue
            x_px, y_px = inv_transform * (float(pt[0]), float(pt[1]))
            px.append((float(x_px), float(y_px)))
        # drop GeoJSON's closing duplicate point so walls don't double up
        if len(px) >= 2 and px[0] == px[-1]:
            px.pop()
        if len(px) >= 3:
            out.append(px)
    return out


def _signed_area(ring: list[tuple[float, float]]) -> float:
    """Shoelace signed area. Positive = CCW, negative = CW."""
    n = len(ring)
    total = 0.0
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _earclip_triangulate(ring: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon (no holes) by ear clipping. Returns
    triangles as triples of indices into the input ring. Handles concave
    L/U-shaped polygons correctly (no external spokes), unlike a centroid
    fan. Falls back to fan triangulation on degenerate inputs.
    """
    n = len(ring)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Ensure CCW winding so the convexity test is well-defined.
    work = ring if _signed_area(ring) > 0 else list(reversed(ring))
    # Map back: if reversed, indices need translating to original space at end.
    reversed_input = work is not ring
    indices = list(range(n))

    def _cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _point_in_tri(px: float, py: float, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
        d1 = _cross(px, py, ax, ay, bx, by)
        d2 = _cross(px, py, bx, by, cx, cy)
        d3 = _cross(px, py, cx, cy, ax, ay)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    triangles: list[tuple[int, int, int]] = []
    safety = n * n  # bail on degenerate / self-intersecting inputs
    while len(indices) > 3 and safety > 0:
        safety -= 1
        clipped = False
        m = len(indices)
        for k in range(m):
            i_prev = indices[(k - 1) % m]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]
            ax, ay = work[i_prev]
            bx, by = work[i_curr]
            cx, cy = work[i_next]
            # Convex corner? (CCW means cross > 0)
            if _cross(ax, ay, bx, by, cx, cy) <= 0:
                continue
            # No other vertex inside this candidate ear?
            ok = True
            for other in indices:
                if other in (i_prev, i_curr, i_next):
                    continue
                px, py = work[other]
                if _point_in_tri(px, py, ax, ay, bx, by, cx, cy):
                    ok = False
                    break
            if ok:
                triangles.append((i_prev, i_curr, i_next))
                indices.pop(k)
                clipped = True
                break
        if not clipped:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    elif len(indices) > 3:
        # Degenerate polygon — fall back to fan from indices[0]
        for k in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[k], indices[k + 1]))

    if reversed_input:
        # Translate ear-clip indices (in reversed ring space) back to original.
        return [(n - 1 - a, n - 1 - b, n - 1 - c) for a, b, c in triangles]
    return triangles


def _extrude_ring(
    ring_px: list[tuple[float, float]],
    ground_z: float,
    roof_z: float,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    color: tuple[int, int, int] = _LOD1_DEFAULT_COLOR,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    n = len(ring_px)
    base = len(vertices)
    # bottom ring (z = ground)
    for x, y in ring_px:
        vertices.append((x, y, ground_z))
        if colors is not None:
            colors.append(color)
    # top ring (z = roof)
    for x, y in ring_px:
        vertices.append((x, y, roof_z))
        if colors is not None:
            colors.append(color)
    # wall quads, outward-facing (CCW when viewed from outside assuming
    # GeoJSON rings are CCW exterior — near enough for visualization).
    for i in range(n):
        j = (i + 1) % n
        b0, b1 = base + i, base + j
        t0, t1 = base + n + i, base + n + j
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
    # Roof cap: ear-clip triangulation on the top ring. Handles L/U-shaped
    # buildings correctly without the external "spokes" a centroid fan
    # produces when the centroid lands outside the polygon (e.g., reflex
    # corners on a row-house U-courtyard).
    top_offset = base + n
    for ti, tj, tk in _earclip_triangulate(ring_px):
        faces.append((top_offset + ti, top_offset + tj, top_offset + tk))


def _build_lod1_mesh(
    features: list[dict[str, Any]],
    transform: Any,
    lidar_heights: Any,
    ground_z: float,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]],
    Any,
    dict[str, int],
    list[tuple[int, int, int]],
]:
    import numpy as np
    from rasterio.features import rasterize

    heights = np.asarray(lidar_heights)
    H, W = int(heights.shape[0]), int(heights.shape[1])
    inv = ~transform

    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    # Union of every polygon we actually extruded into the mesh — this is what
    # the QA stage IoUs against the SAM2 building mask. Demolished/skipped
    # features are intentionally excluded so the union reflects the mesh
    # geometry, not the input footprints.
    footprint_mask = np.zeros((H, W), dtype=bool)
    stats = {"count": 0, "skipped_empty": 0, "default_height": 0, "skipped_demolished": 0}

    for feat in features:
        props = feat.get("properties") or {}
        change_type = str(props.get("change_type") or "").strip().lower()
        if change_type == "demolished":
            stats["skipped_demolished"] += 1
            continue
        building_color = _LOD1_BUILDING_COLOR.get(change_type, _LOD1_DEFAULT_COLOR)
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            mask = rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
        except Exception:
            stats["skipped_empty"] += 1
            continue
        if not mask.any():
            stats["skipped_empty"] += 1
            continue

        cell = heights[mask]
        finite = cell[np.isfinite(cell)]
        if finite.size == 0:
            roof_z = float(ground_z) + _LOD1_DEFAULT_HEIGHT_M
            stats["default_height"] += 1
        else:
            roof_z = float(np.percentile(finite, _LOD1_ROOF_PERCENTILE))
            if roof_z <= float(ground_z):
                roof_z = float(ground_z) + _LOD1_DEFAULT_HEIGHT_M
                stats["default_height"] += 1

        rings_px = _feature_rings_pixel(geom, inv)
        if not rings_px:
            stats["skipped_empty"] += 1
            continue
        for ring_px in rings_px:
            _extrude_ring(
                ring_px,
                float(ground_z),
                roof_z,
                vertices,
                faces,
                color=building_color,
                colors=colors,
            )
        footprint_mask |= mask
        stats["count"] += 1

    return vertices, faces, footprint_mask, stats, colors


def _write_ply_ascii(
    out_path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None and len(colors) == len(vertices)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        if has_color:
            for (x, y, z), (r, g, b) in zip(vertices, colors):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
        else:
            for x, y, z in vertices:
                f.write(f"{x} {y} {z}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


# ----------------------------------------------------------------------
# Legacy heightfield (fallback only)
# ----------------------------------------------------------------------


def _write_height_mesh_ply(height_map, out_path: Path, *, max_dim: int = 256) -> None:
    import numpy as np

    m = np.asarray(height_map).astype(np.float32)
    if m.ndim != 2:
        raise ValueError("height_map must be a 2D array")

    h, w = m.shape
    step = max(1, int(max(h, w) // max_dim))
    mm = m[::step, ::step]
    hh, ww = mm.shape

    vertices = []
    for y in range(hh):
        for x in range(ww):
            z = float(mm[y, x])
            vertices.append((float(x), float(y), z))

    faces = []

    def vid(x: int, y: int) -> int:
        return y * ww + x

    for y in range(hh - 1):
        for x in range(ww - 1):
            v00 = vid(x, y)
            v10 = vid(x + 1, y)
            v01 = vid(x, y + 1)
            v11 = vid(x + 1, y + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(f"{x} {y} {z}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


# ----------------------------------------------------------------------
# Stage entry
# ----------------------------------------------------------------------


def stage_reconstruct(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "mesh.ply"
    transform = ctx.get("orthophoto_transform")
    lidar_heights = ctx.get("lidar_heights")
    lidar_ground_z = ctx.get("lidar_ground_z")
    change_path = ctx.get("change_path")

    # LOD1 path: emit one extrusion per non-demolished classified footprint +
    # accepted 'added' component. Requires everything stage_refine +
    # stage_change normally produces on a real NYC run.
    if (
        transform is not None
        and lidar_heights is not None
        and lidar_ground_z is not None
        and change_path is not None
        and Path(change_path).exists()
    ):
        features = _load_change_features(Path(change_path))
        if features:
            vertices, faces, footprint_mask, stats, colors = _build_lod1_mesh(
                features, transform, lidar_heights, float(lidar_ground_z)
            )
            if vertices and faces:
                _write_ply_ascii(out_path, vertices, faces, colors=colors)
                summary.qa["mesh_source"] = "lod1"
                summary.qa["mesh_buildings"] = stats["count"]
                summary.qa["mesh_stats"] = dict(stats)
                return {
                    **ctx,
                    "mesh_path": out_path,
                    "mesh_footprint_mask": footprint_mask.astype("uint8"),
                    "mesh_height_source": "lidar",
                }

    # Fallback: legacy heightfield over the whole ortho. Kept for tests
    # and non-NYC paths where change.geojson / LiDAR may be missing.
    mask = ctx.get("refined_mask", ctx.get("mask"))
    if mask is None:
        raise RuntimeError("reconstruct stage requires a segmentation mask")

    lidar_debug: dict[str, Any] = {}
    height_map, footprint_mask, source = build_height_map_from_lidar(
        mask,
        work_dir / "lidar.las",
        transform,
        dst_crs=ctx.get("orthophoto_crs"),
        debug=lidar_debug,
    )
    summary.qa["lidar_debug"] = lidar_debug
    summary.qa["mesh_source"] = "heightfield"
    _write_height_mesh_ply(height_map, out_path)

    return {
        **ctx,
        "mesh_path": out_path,
        "mesh_footprint_mask": footprint_mask.astype("uint8"),
        "mesh_height_source": source,
    }
