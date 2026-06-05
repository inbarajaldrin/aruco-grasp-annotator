"""
Job specification + result models for the deterministic CAD->outputs pipeline.

A *job* is one orchestrator run that produces one output folder. It bundles >= 1
assembly; each assembly becomes a per-assembly subfolder under the job's output dir. The
``JobSpec`` is the single source of truth for a run and is written verbatim into the job
folder as part of its provenance manifest. It is also the schema a future MCP tool layer
will wrap, so keep it declarative and JSON-round-trippable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Canonical stage order. The orchestrator enforces the data-dependency DAG within this
# order (markers before png/pdf; wireframe + markers before grasp; symmetry independent).
Stage = Literal["wireframe", "markers", "markers_png", "markers_pdf", "symmetry", "grasp"]
ALL_STAGES: tuple[Stage, ...] = (
    "wireframe",
    "markers",
    "markers_png",
    "markers_pdf",
    "symmetry",
    "grasp",
)

MarkerFamilyName = Literal["aruco", "apriltag"]
RunMode = Literal["rebuild", "verify", "skip_existing"]
StageStatus = Literal["ok", "skip", "fail"]


class AssemblySpec(BaseModel):
    """One assembly in a job: a name + the path to its components config.

    The config is an ``fmb_assembly*.json``-shaped file (``components[]`` with
    name/type/subtype/position/rotation/assembly_order). Object meshes are resolved by
    component name from the job-level ``models_dir``.
    """

    name: str
    config: Path

    @field_validator("config")
    @classmethod
    def _config_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"assembly config not found: {v}")
        return v


class JobSpec(BaseModel):
    """Everything one orchestrator run needs. Paths are caller-supplied variables."""

    output_dir: Path
    assemblies: list[AssemblySpec]
    models_dir: Path

    marker_family: MarkerFamilyName = "aruco"
    border_width: float | None = None  # None -> the family default

    stages: list[Stage] = Field(default_factory=lambda: list(ALL_STAGES))
    skip: list[Stage] = Field(default_factory=list)

    mode: RunMode = "rebuild"
    bundle_models: bool = False

    @field_validator("models_dir")
    @classmethod
    def _models_dir_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"models_dir not found: {v}")
        return v

    def active_stages(self) -> list[Stage]:
        """Requested stages minus skipped, in canonical order."""
        requested = set(self.stages) - set(self.skip)
        return [s for s in ALL_STAGES if s in requested]

    @classmethod
    def from_json_file(cls, path: str | Path) -> "JobSpec":
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)

    def to_json(self) -> str:
        # mode="json" renders Path as str so the manifest is plain JSON.
        return json.dumps(self.model_dump(mode="json"), indent=2)


# --- result models (returned by run_job; also serialized into the manifest) ----------


class StageResult(BaseModel):
    stage: Stage
    object: str | None = None  # None for assembly-level summaries
    status: StageStatus
    detail: str = ""


class AssemblyResult(BaseModel):
    name: str
    marker_dictionary: str | None = None
    total_markers: int = 0
    stages: list[StageResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.stages if s.status == "fail")


class JobResult(BaseModel):
    output_dir: Path
    assemblies: list[AssemblyResult] = Field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(a.failed for a in self.assemblies)

    @property
    def ok(self) -> bool:
        return self.failed == 0
