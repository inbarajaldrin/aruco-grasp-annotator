"""
Job provenance (Q6): make a job folder self-describing and reproducible.

For every assembly the orchestrator snapshots its inputs under ``<asm_out>/input/``:
a verbatim copy of the assembly config, and a content hash (sha256) of each referenced
mesh. ``bundle_models`` additionally copies the meshes in, turning the job folder into a
portable, dependency-free archive. The job-level ``manifest.json`` then ties the resolved
JobSpec, the per-assembly input snapshots, and the run results together — so months later
you can prove exactly which geometry + parameters produced these outputs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .spec import JobResult, JobSpec
from .stages import find_model, walk_order


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_assembly_inputs(
    asm_out: Path,
    config_path: Path,
    components: list[dict[str, Any]],
    models_dir: Path,
    *,
    bundle_models: bool,
) -> dict[str, Any]:
    """Copy the config, hash (and optionally bundle) the meshes. Returns a snapshot dict."""
    input_dir = asm_out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, input_dir / "config.json")

    meshes: dict[str, Any] = {}
    bundle_dir = input_dir / "models"
    if bundle_models:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    for comp in walk_order(components):
        name = comp["name"]
        mesh_path = find_model(models_dir, name)
        if mesh_path is None:
            meshes[name] = {"path": None, "sha256": None}
            continue
        digest = sha256_file(mesh_path)
        entry = {"path": str(mesh_path), "sha256": digest}
        if bundle_models:
            dest = bundle_dir / mesh_path.name
            shutil.copyfile(mesh_path, dest)
            # carry sibling .mtl for textured .obj meshes so the bundle is complete
            mtl = mesh_path.with_suffix(".mtl")
            if mtl.exists():
                shutil.copyfile(mtl, bundle_dir / mtl.name)
            entry["bundled"] = str(dest.relative_to(asm_out))
        meshes[name] = entry

    return {
        "config": str(config_path),
        "config_copy": str((input_dir / "config.json").relative_to(asm_out)),
        "bundled_models": bool(bundle_models),
        "meshes": meshes,
    }


def write_manifest(
    output_dir: Path,
    spec: JobSpec,
    result: JobResult,
    input_snapshots: dict[str, dict[str, Any]],
) -> Path:
    """Write ``<output_dir>/manifest.json`` tying spec + inputs + results together."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "spec": json.loads(spec.to_json()),
        "assemblies": [
            {
                "name": a.name,
                "marker_dictionary": a.marker_dictionary,
                "total_markers": a.total_markers,
                "warnings": a.warnings,
                "inputs": input_snapshots.get(a.name, {}),
                "stages": [s.model_dump() for s in a.stages],
                "failed": a.failed,
            }
            for a in result.assemblies
        ],
        "ok": result.ok,
        "failed": result.failed,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
