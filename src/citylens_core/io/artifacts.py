from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

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


def validate_standard_artifacts(work_dir: Path) -> None:
    missing: list[str] = []
    for name in ("preview.png", "change.geojson", "mesh.ply", "run_summary.json"):
        p = work_dir / name
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + ", ".join(missing))


def write_run_summary(summary: PipelineSummary, path: Path, extra: Optional[dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json")
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
