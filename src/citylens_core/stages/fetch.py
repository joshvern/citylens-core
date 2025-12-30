from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..models import CitylensRequest, PipelineSummary


def stage_fetch(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    """Fetch inputs.

    This core package intentionally contains no cloud/provider-specific logic.
    For now, write placeholder imagery that downstream stages can consume.
    """

    ortho_path = work_dir / "orthophoto.png"
    baseline_path = work_dir / "baseline.png"

    if not ortho_path.exists():
        Image.new("RGB", (512, 512), color=(120, 120, 120)).save(ortho_path)
    if not baseline_path.exists():
        Image.new("RGB", (512, 512), color=(110, 110, 110)).save(baseline_path)

    return {**ctx, "orthophoto_path": ortho_path, "baseline_path": baseline_path}
