from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.artifacts import write_placeholder_ply
from ..models import CitylensRequest, PipelineSummary


def stage_reconstruct(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "mesh.ply"

    try:
        import open3d as o3d  # optional

        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=True)
    except Exception as e:
        summary.warn(f"mesh: {type(e).__name__}: {e}; writing placeholder mesh")
        write_placeholder_ply(out_path)

    return {**ctx, "mesh_path": out_path}
