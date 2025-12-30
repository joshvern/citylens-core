from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import CitylensRequest, PipelineSummary


def stage_render(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    out_path = work_dir / "preview.png"

    ortho_path = Path(ctx.get("orthophoto_path", work_dir / "orthophoto.png"))
    img = Image.open(ortho_path).convert("RGBA")
    base = np.array(img).astype(np.uint8)

    mask = ctx.get("mask")
    if mask is None:
        Image.fromarray(base).save(out_path)
        return {**ctx, "preview_path": out_path}

    m = np.asarray(mask).astype(bool)
    overlay = base.copy()
    overlay[m, 0] = 255
    overlay[m, 1] = 0
    overlay[m, 2] = 0
    overlay[m, 3] = 160

    Image.fromarray(overlay).save(out_path)
    return {**ctx, "preview_path": out_path}
