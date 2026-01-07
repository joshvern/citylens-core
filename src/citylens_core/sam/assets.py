from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import importlib.util

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


def assets_root() -> Path:
    """Base directory for runtime assets (configs/, weights/).

    Prefer an explicit env override, otherwise use the current working directory.
    """

    root = os.getenv("CITYLENS_ASSETS_ROOT", "").strip()
    if root:
        return Path(root).resolve()
    return Path.cwd().resolve()


def ensure_sam2_assets(cfg_path: Path, ckpt_path: Path) -> None:
    if not str(cfg_path).strip() or str(cfg_path).strip() in (".", "./"):
        raise Sam2AssetsMissingError("SAM2 config path is empty")
    if not str(ckpt_path).strip() or str(ckpt_path).strip() in (".", "./"):
        raise Sam2AssetsMissingError("SAM2 checkpoint path is empty")

    # SAM2 model config is typically resolved by Hydra from the installed `sam2` package.
    # Only treat cfg_path as a filesystem path when it is absolute.
    cfg: Path | None = cfg_path if cfg_path.is_absolute() else None

    # Checkpoint must be a real file on disk; we support relative paths resolved from the assets root.
    root = assets_root()
    ckpt = ckpt_path if ckpt_path.is_absolute() else (root / ckpt_path)

    if cfg is None:
        # Best-effort validation: if `sam2` is installed, verify the config exists within the package.
        spec = importlib.util.find_spec("sam2")
        if spec and spec.origin:
            pkg_root = Path(spec.origin).resolve().parent
            candidate = (pkg_root / str(cfg_path)).resolve()
            if not candidate.exists() or not candidate.is_file():
                # Some installs expect config names like "sam2.1/sam2.1_hiera_s.yaml" (without a leading "configs/").
                alt = (pkg_root / "configs" / str(cfg_path)).resolve()
                if not alt.exists() or not alt.is_file():
                    raise Sam2AssetsMissingError(
                        "SAM2 config not found in installed sam2 package: "
                        f"{cfg_path} (also tried {alt.relative_to(pkg_root) if alt.is_absolute() else alt})."
                    )

    missing = []
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
    root = assets_root()
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
