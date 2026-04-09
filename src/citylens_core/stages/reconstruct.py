from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary
from ..io.lidar import build_height_map_from_lidar


def _write_height_mesh_ply(height_map, out_path: Path, *, max_dim: int = 256) -> None:
    import numpy as np

    m = np.asarray(height_map).astype(np.float32)
    if m.ndim != 2:
        raise ValueError("height_map must be a 2D array")

    h, w = m.shape
    step = max(1, int(max(h, w) // max_dim))
    mm = m[::step, ::step]
    hh, ww = mm.shape

    # vertices: grid points (x,y) with z from mask
    vertices = []
    for y in range(hh):
        for x in range(ww):
            z = float(mm[y, x])
            vertices.append((float(x), float(y), z))

    # faces: two triangles per quad
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


def stage_reconstruct(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "mesh.ply"

    mask = ctx.get("refined_mask", ctx.get("mask"))
    if mask is None:
        raise RuntimeError("reconstruct stage requires a segmentation mask")

    height_map, footprint_mask, source = build_height_map_from_lidar(
        mask,
        work_dir / "lidar.las",
        ctx.get("orthophoto_transform"),
    )
    _write_height_mesh_ply(height_map, out_path)

    return {
        **ctx,
        "mesh_path": out_path,
        "mesh_footprint_mask": footprint_mask.astype("uint8"),
        "mesh_height_source": source,
    }
