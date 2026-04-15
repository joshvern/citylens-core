from __future__ import annotations

from pathlib import Path


def test_relative_sam2_checkpoint_resolves_from_assets_root_outside_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    from citylens_core.sam.assets import ensure_sam2_assets

    runtime_assets = tmp_path / "runtime-assets"
    checkpoint = runtime_assets / "weights" / "sam2.1_hiera_small.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")

    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()

    monkeypatch.setenv("CITYLENS_ASSETS_ROOT", str(runtime_assets))
    monkeypatch.chdir(elsewhere)

    ensure_sam2_assets(
        Path("configs/sam2.1/sam2.1_hiera_s.yaml"),
        Path("weights/sam2.1_hiera_small.pt"),
    )
