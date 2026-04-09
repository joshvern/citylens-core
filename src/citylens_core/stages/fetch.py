from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from rasterio.errors import RasterioIOError

from ..models import CitylensRequest, PipelineSummary


def _download_url(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _raster_metadata(path: Path) -> dict[str, Any]:
    try:
        import rasterio

        with rasterio.open(path) as src:
            crs = src.crs.to_string() if src.crs else None
            transform = src.transform if src.crs else None
            return {"crs": crs, "transform": transform}
    except (RasterioIOError, OSError, ValueError):
        return {"crs": None, "transform": None}
    except Exception:
        return {"crs": None, "transform": None}


def _resolve_input(
    *,
    url: str | None,
    local_path: Path | None,
    default_path: Path,
    label: str,
) -> Path:
    if url:
        if not default_path.exists():
            _download_url(url, default_path)
        return default_path

    if local_path is not None:
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Missing required input data: {local}")
        return local

    if not default_path.exists():
        raise FileNotFoundError(f"Missing required input data: {default_path} ({label})")
    return default_path


def stage_fetch(
    request: CitylensRequest,
    work_dir: Path,
    ctx: dict[str, Any],
    summary: PipelineSummary,
) -> dict[str, Any]:
    """Resolve input paths.

    citylens-core does not synthesize placeholder data. Inputs must either be provided
    explicitly on the request (preferred), fetched from explicit URLs, or pre-populated
    in the work_dir.
    """

    ortho_path = _resolve_input(
        url=request.orthophoto_url,
        local_path=request.orthophoto_path,
        default_path=work_dir / "orthophoto.png",
        label="orthophoto",
    )
    ortho_meta = _raster_metadata(ortho_path)

    out: dict[str, Any] = {
        **ctx,
        "orthophoto_path": ortho_path,
        "orthophoto_crs": ortho_meta["crs"],
        "orthophoto_transform": ortho_meta["transform"],
    }

    want_change = "change" in {str(o).strip().lower() for o in (request.outputs or []) if str(o).strip()}
    if want_change:
        baseline_path = _resolve_input(
            url=request.baseline_url,
            local_path=request.baseline_path,
            default_path=work_dir / "baseline.png",
            label="baseline",
        )
        baseline_meta = _raster_metadata(baseline_path)
        out.update(
            {
                "baseline_path": baseline_path,
                "baseline_crs": baseline_meta["crs"],
                "baseline_transform": baseline_meta["transform"],
            }
        )

    return out
