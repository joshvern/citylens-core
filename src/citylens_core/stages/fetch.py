from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import CitylensRequest, PipelineSummary


def stage_fetch(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    """Resolve input paths.

    citylens-core does not synthesize placeholder data. Inputs must either be provided
    explicitly on the request (preferred) or pre-populated in the work_dir.
    """

    ortho_path = Path(request.orthophoto_path) if request.orthophoto_path else (work_dir / "orthophoto.png")
    baseline_path = (
        Path(request.baseline_path) if request.baseline_path else (work_dir / "baseline.png")
    )

    missing: list[str] = []
    if not ortho_path.exists():
        missing.append(str(ortho_path))
    if not baseline_path.exists():
        missing.append(str(baseline_path))
    if missing:
        raise FileNotFoundError("Missing required input data: " + ", ".join(missing))

    return {**ctx, "orthophoto_path": ortho_path, "baseline_path": baseline_path}
