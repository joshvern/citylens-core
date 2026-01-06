from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CitylensRequest(BaseModel):
    address: str
    aoi_radius_m: int = 250
    imagery_year: int = 2024
    baseline_year: int = 2017
    segmentation_backend: Literal["unet", "smp", "sam2"] = "sam2"
    sam2_cfg: Optional[str] = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_checkpoint: Optional[str] = "weights/sam2.1_hiera_small.pt"
    orthophoto_path: Optional[Path] = None
    baseline_path: Optional[Path] = None
    outputs: list[str] = Field(default_factory=lambda: ["previews", "change", "mesh"])
    notes: Optional[str] = None


class Artifact(BaseModel):
    name: str
    path: Path
    sha256: str
    size_bytes: int


class PipelineSummary(BaseModel):
    request: CitylensRequest
    work_dir: Path
    started_at: datetime
    finished_at: Optional[datetime] = None
    ok: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    missing_paths: list[str] = Field(default_factory=list)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stage_status: dict[str, str] = Field(default_factory=dict)

    def warn(self, msg: str) -> None:
        self.warnings.append(str(msg))

    def error(self, msg: str) -> None:
        self.errors.append(str(msg))
        self.ok = False

    def fail(self, *, error_code: str, error_message: str, missing_paths: Optional[list[str]] = None) -> None:
        self.ok = False
        self.error_code = str(error_code)
        self.error_message = str(error_message)
        if missing_paths:
            self.missing_paths = [str(p) for p in missing_paths]
        self.error(error_message)
