"""
run_job: sequence the 6 stages over a job's assemblies into one output folder.

Because every stage is an in-process core call (Q2) and IDs/dictionaries are pure metadata
(Q7/Q11), the orchestrator is genuinely just sequencing + bookkeeping: walk assemblies,
run the active stages in canonical DAG order, collect per-object/stage status fail-soft,
and write the provenance manifest. Three run modes:

  * rebuild      - overwrite outputs in place (default).
  * skip_existing- only generate what's missing (fast iteration).
  * verify       - regenerate into a temp dir and canonical-diff against the existing job
                   folder; write nothing. The determinism regression guard: it answers
                   "did my code change alter the outputs?" (volatile timestamps stripped,
                   floats compared to 1e-6 so the ~7e-9 wireframe base-2 drift is a match).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

try:  # headless rendering for the grasp stage's matplotlib/open3d usage
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # noqa: BLE001
    pass

from . import stages
from .manifest import snapshot_assembly_inputs, write_manifest
from .spec import ALL_STAGES, AssemblyResult, JobResult, JobSpec, Stage, StageResult

# Stage -> output subdir(s) relative to an assembly dir, for verify-mode tree comparison.
_VOLATILE_KEYS = {"exported_at"}
_FLOAT_DECIMALS = 6  # collapses the ~7e-9 base-2 float drift seen in wireframe export

# Data-dependency prerequisites: a requested downstream stage pulls in any prerequisite
# whose outputs aren't already on disk, so e.g. `--stages grasp` on a fresh job auto-runs
# wireframe + markers, while on an already-built job it runs grasp alone.
_PREREQS: dict[Stage, tuple[Stage, ...]] = {
    "markers_png": ("markers",),
    "markers_pdf": ("markers",),
    "grasp": ("markers", "wireframe"),
}


def _present(stage: Stage, asm_out: Path, family: str) -> bool:
    """Is this prerequisite stage's output already on disk for the assembly?"""
    if stage == "markers":
        d = asm_out / family
        return d.is_dir() and any(d.glob("*_aruco.json"))
    if stage == "wireframe":
        d = asm_out / "wireframe"
        return d.is_dir() and any(d.glob("*_wireframe.json"))
    return False


def _resolve_stages(active: list[Stage], asm_out: Path, family: str) -> list[Stage]:
    """Expand the active set with any missing prerequisites, in canonical order."""
    effective = set(active)
    for dep, reqs in _PREREQS.items():
        if dep in effective:
            for req in reqs:
                if req not in effective and not _present(req, asm_out, family):
                    effective.add(req)
    return [s for s in ALL_STAGES if s in effective]


# --- public entry point ---------------------------------------------------------------


def run_job(spec: JobSpec) -> JobResult:
    result = JobResult(output_dir=spec.output_dir)
    snapshots: dict[str, dict[str, Any]] = {}

    for asm in spec.assemblies:
        components = stages.load_components(asm.config)
        if spec.mode == "verify":
            result.assemblies.append(_verify_assembly(spec, asm.name, components))
            continue

        asm_out = spec.output_dir / asm.name
        asm_out.mkdir(parents=True, exist_ok=True)
        result.assemblies.append(_run_assembly(spec, asm.name, components, asm_out))
        snapshots[asm.name] = snapshot_assembly_inputs(
            asm_out, asm.config, components, spec.models_dir, bundle_models=spec.bundle_models
        )

    if spec.mode != "verify":
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(spec.output_dir, spec, result, snapshots)
    return result


# --- per-assembly run -----------------------------------------------------------------


def _run_assembly(
    spec: JobSpec, name: str, components: list[dict[str, Any]], asm_out: Path
) -> AssemblyResult:
    ares = AssemblyResult(name=name)
    active = _resolve_stages(spec.active_stages(), asm_out, spec.marker_family)
    skip = spec.mode == "skip_existing"
    models = spec.models_dir

    if "wireframe" in active:
        ares.stages += stages.stage_wireframe(components, models, asm_out, skip_existing=skip)

    if "markers" in active:
        res, chosen, total, warns = stages.stage_markers(
            components, models, asm_out,
            family=spec.marker_family, border_width=spec.border_width, skip_existing=skip,
        )
        ares.stages += res
        ares.marker_dictionary = chosen
        ares.total_markers = total
        ares.warnings += warns

    if "markers_png" in active:
        ares.stages += stages.stage_marker_pngs(asm_out, family=spec.marker_family, skip_existing=skip)

    if "markers_pdf" in active:
        ares.stages += stages.stage_marker_pdfs(asm_out, family=spec.marker_family, skip_existing=skip)

    if "symmetry" in active:
        ares.stages += stages.stage_symmetry(components, models, asm_out, skip_existing=skip)

    if "grasp" in active:
        ares.stages += stages.stage_grasp(components, models, asm_out, family=spec.marker_family, skip_existing=skip)

    return ares


# --- verify mode: regenerate to temp, canonical-diff against the existing job folder ---


def _verify_assembly(spec: JobSpec, name: str, components: list[dict[str, Any]]) -> AssemblyResult:
    existing = spec.output_dir / name
    ares = AssemblyResult(name=name)
    if not existing.exists():
        ares.stages.append(StageResult(stage="markers", object=None, status="fail",
                                       detail=f"nothing to verify against: {existing} missing"))
        return ares

    tmp_root = Path(tempfile.mkdtemp(prefix="verify_job_"))
    try:
        fresh = tmp_root / name
        fresh.mkdir(parents=True, exist_ok=True)
        # Force a full rebuild into the temp dir (no skip), reusing the normal path.
        rebuild_spec = spec.model_copy(update={"mode": "rebuild"})
        _run_assembly(rebuild_spec, name, components, fresh)
        ares.stages = _compare_trees(existing, fresh)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return ares


def _stage_of(rel: Path) -> str:
    """Map an output file's top subdir to a Stage label for verify reporting."""
    top = rel.parts[0] if rel.parts else ""
    if top == "wireframe":
        return "wireframe"
    if top.endswith("_pngs"):
        return "markers_png"
    if top.endswith("_pdfs"):
        return "markers_pdf"
    if top == "symmetry":
        return "symmetry"
    if top == "grasp_points":
        return "grasp"
    return "markers"  # the family folder (aruco/apriltag)


_OBJ_SUFFIXES = ("_aruco", "_wireframe", "_symmetry", "_grasp_points")


def _obj_of(rel: Path) -> str:
    """Recover the object name from an output filename, stripping only the TERMINAL token
    (so an object literally named e.g. ``left_aruco_mount`` is not mangled)."""
    stem = rel.name.split("_marker_")[0] if "_marker_" in rel.name else rel.stem
    for suf in _OBJ_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _rel_files(root: Path) -> set[Path]:
    """Relative paths of output files under an assembly dir, excluding the input/ snapshot."""
    return {
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts and p.relative_to(root).parts[0] != "input"
    }


def _compare_trees(existing: Path, fresh: Path) -> list[StageResult]:
    """Canonical per-file comparison. JSON: volatile-stripped + float-tolerant. PNG: bytes.
    PDFs are reported skip (reportlab embeds a creation date); the input/ snapshot is ignored.
    Files present in ``existing`` but NOT regenerated (stale) are flagged when their stage was
    part of this run — so a stale leftover can't hide behind an all-green verify."""
    results: list[StageResult] = []
    fresh_files = _rel_files(fresh)
    existing_files = _rel_files(existing)
    fresh_stages = {_stage_of(r) for r in fresh_files}

    for rel in sorted(fresh_files):
        stage = _stage_of(rel)
        obj = _obj_of(rel)
        ex = existing / rel
        fr = fresh / rel
        if rel.suffix == ".pdf":
            results.append(StageResult(stage=stage, object=obj, status="skip", detail="pdf non-deterministic"))
            continue
        if not ex.exists():
            results.append(StageResult(stage=stage, object=obj, status="fail", detail="missing in existing job"))
            continue
        same = _json_equal(ex, fr) if rel.suffix == ".json" else ex.read_bytes() == fr.read_bytes()
        results.append(StageResult(stage=stage, object=obj, status="ok" if same else "fail",
                                   detail="" if same else "differs from existing"))

    # Stale outputs: in existing, belong to a stage that ran, but weren't regenerated.
    for rel in sorted(existing_files - fresh_files):
        stage = _stage_of(rel)
        if stage not in fresh_stages or rel.suffix == ".pdf":
            continue
        results.append(StageResult(stage=stage, object=_obj_of(rel), status="fail",
                                   detail="stale in existing job (not regenerated)"))
    return results


def _json_equal(a: Path, b: Path) -> bool:
    import json

    return _canon(json.loads(a.read_text())) == _canon(json.loads(b.read_text()))


def _canon(obj: Any) -> Any:
    """Strip volatile keys, round floats — so reruns compare equal despite timestamps/drift."""
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, _FLOAT_DECIMALS)
    return obj
