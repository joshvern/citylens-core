from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


def stage_resolve(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    return {**ctx, "work_dir": work_dir}
