from __future__ import annotations

from pathlib import Path
from typing import Literal

import requests


class SamAssetsError(RuntimeError):
    pass


class Sam2AssetsMissingError(SamAssetsError):
    pass


SAM2_YAML_BASE = "https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1"
SAM2_CKPT_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


def _repo_root() -> Path:
    # repo root when developing/editable install: citylens-core/
    return Path(__file__).resolve().parents[3]


def ensure_sam2_assets(cfg_path: Path, ckpt_path: Path) -> None:
    if not str(cfg_path).strip() or str(cfg_path).strip() in (".", "./"):
        raise Sam2AssetsMissingError("SAM2 config path is empty")
    if not str(ckpt_path).strip() or str(ckpt_path).strip() in (".", "./"):
        raise Sam2AssetsMissingError("SAM2 checkpoint path is empty")

    cfg = cfg_path if cfg_path.is_absolute() else (_repo_root() / cfg_path)
    ckpt = ckpt_path if ckpt_path.is_absolute() else (_repo_root() / ckpt_path)
    missing = []
    if not cfg.exists() or not cfg.is_file():
        missing.append(str(cfg))
    if not ckpt.exists() or not ckpt.is_file():
        missing.append(str(ckpt))
    if missing:
        raise Sam2AssetsMissingError(
            "SAM2 assets missing: " + ", ".join(missing) + ". Run `make sam2-assets` to download."
        )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_sam2_assets(size: Literal["small", "large"] = "small") -> None:
    root = _repo_root()
    (root / "configs" / "sam2.1").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)

    if size == "small":
        _download(f"{SAM2_YAML_BASE}/sam2.1_hiera_s.yaml", root / "configs/sam2.1/sam2.1_hiera_s.yaml")
        _download(f"{SAM2_CKPT_BASE}/sam2.1_hiera_small.pt", root / "weights/sam2.1_hiera_small.pt")
        return
    if size == "large":
        _download(f"{SAM2_YAML_BASE}/sam2.1_hiera_l.yaml", root / "configs/sam2.1/sam2.1_hiera_l.yaml")
        _download(f"{SAM2_CKPT_BASE}/sam2.1_hiera_large.pt", root / "weights/sam2.1_hiera_large.pt")
        return
    raise ValueError(f"Unsupported size: {size}")
