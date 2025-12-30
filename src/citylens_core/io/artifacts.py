from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import Artifact, PipelineSummary


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_artifact_metadata(name: str, path: Path) -> Artifact:
    st = path.stat()
    return Artifact(name=name, path=path, sha256=sha256_file(path), size_bytes=int(st.st_size))


def write_placeholder_preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), color=(30, 30, 30)).save(path)


def write_placeholder_change_geojson(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": []}
    path.write_text(json.dumps(fc, indent=2))


def write_placeholder_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid ASCII PLY with no vertices/faces
    content = """ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n"""
    path.write_text(content)


def ensure_standard_artifacts(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)

    preview = work_dir / "preview.png"
    change = work_dir / "change.geojson"
    mesh = work_dir / "mesh.ply"

    if not preview.exists():
        write_placeholder_preview(preview)
    if not change.exists():
        write_placeholder_change_geojson(change)
    if not mesh.exists():
        write_placeholder_ply(mesh)


def write_run_summary(summary: PipelineSummary, path: Path, extra: Optional[dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json")
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
