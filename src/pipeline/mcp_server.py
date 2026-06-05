"""
FastMCP server exposing the CAD -> outputs pipeline as agent-callable tools.

Reference patterns mirrored from ../ros-mcp-server/server.py: FastMCP instance, @mcp.tool()
functions with Annotated[..., Field(description=...)] args, Pydantic BaseModel return types
with a "Returns:" docstring block, Literal type aliases, and mcp.run(transport="stdio").

Addressing follows the locked design: an `assets/` bundle (default ./assets, override with
$ARUCO_ASSETS_ROOT) with input/ (models, parts.json catalog, assemblies/ recipes) and
output/ (per-assembly stage outputs). Tools take an `assembly` (recipe name) and optional
`object` filter — never raw paths. Generation is deterministic + overwrite-in-place.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .stages import (
    stage_grasp,
    stage_marker_pdfs,
    stage_marker_pngs,
    stage_markers,
    stage_symmetry,
    stage_wireframe,
)

# ── assets bundle resolution ──────────────────────────────────────────────────
ASSETS_ROOT = Path(os.getenv("ARUCO_ASSETS_ROOT", "./assets")).expanduser()

Status = Literal["ok", "skip", "fail"]
MarkerFamily = Literal["aruco", "apriltag"]
ExportFormat = Literal["png", "pdf", "both"]
GraspAxis = Literal["current", "+x", "-x", "+y", "-y", "+z", "-z"]
GripperAxes = Literal["x", "y", "both"]

# Approach axis -> the marker face that points "up" for the top-down render.
_AXIS_FACE = {"current": "top", "+z": "top", "-z": "bottom",
              "+x": "right", "-x": "left", "+y": "front", "-y": "back"}
# Approach axis -> bounding-box index for the object's thickness along that axis.
_AXIS_INDEX = {"current": 2, "+z": 2, "-z": 2, "+x": 0, "-x": 0, "+y": 1, "-y": 1}
_MODEL_EXTS = (".obj", ".stl", ".ply")


def _input_dir() -> Path:
    return ASSETS_ROOT / "input"


def _recipe_path(assembly: str) -> Path:
    return _input_dir() / "assemblies" / f"{assembly}.json"


def _output_dir(assembly: str) -> Path:
    return ASSETS_ROOT / "output" / assembly


def _load_catalog() -> dict[str, dict]:
    """parts.json: {name: {type, subtype?}} — the part catalog (may be absent)."""
    p = _input_dir() / "parts.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_components(assembly: str) -> list[dict]:
    """Build the stage-ready component dicts for an assembly recipe.

    Accepts the new recipe shape ({assembly, parts:[{name, assembly_order, type?, subtype?}]})
    and, for backward compatibility, a legacy fmb config ({components:[{name,type,subtype,
    assembly_order,...}]}). type/subtype come from the recipe inline if present, else the
    parts.json catalog. Raises FileNotFoundError if the recipe is missing.
    """
    rp = _recipe_path(assembly)
    if not rp.exists():
        raise FileNotFoundError(f"no recipe for assembly {assembly!r} at {rp}")
    recipe = json.loads(rp.read_text())

    if "components" in recipe:  # legacy fmb config — already stage-ready
        return list(recipe["components"])

    catalog = _load_catalog()
    components: list[dict] = []
    for part in recipe.get("parts", []):
        name = part["name"]
        cat = catalog.get(name, {})
        components.append(
            {
                "name": name,
                "type": part.get("type", cat.get("type", "object")),
                "subtype": part.get("subtype", cat.get("subtype")),
                "assembly_order": part.get("assembly_order", 0),
            }
        )
    return components


# ── result models ─────────────────────────────────────────────────────────────
# Shared envelope (ros-mcp convention): one location, bare names, omit-when-empty.
# skipped/failed are Optional and dropped via model_dump(exclude_none=True) when none.
class SkipEntry(BaseModel):
    name: str = Field(description="CAD object name")
    reason: str = Field(description="Why it was skipped, e.g. 'board' or 'no mesh'")


class FailEntry(BaseModel):
    name: str = Field(description="CAD object name")
    error: str = Field(description="Failure detail")


class WireframeResult(BaseModel):
    output_dir: str = Field(description="Directory the files were written to (stated once; encodes the assembly)")
    generated: list[str] = Field(description="Object names written (file = <name>_wireframe.json)")
    skipped: Optional[list[SkipEntry]] = Field(default=None, description="Objects skipped, with reason (omitted if none)")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Objects that failed, with error (omitted if none)")


class SymmetryResult(BaseModel):
    output_dir: str = Field(description="Directory the files were written to (stated once; encodes the assembly)")
    generated: list[str] = Field(description="Object names written (file = <name>_symmetry.json)")
    folds: dict[str, dict[str, int]] = Field(description="Per object: N-fold per axis {x,y,z}; 1 = none, 2 = 2-fold, 6 = hex")
    skipped: Optional[list[SkipEntry]] = Field(default=None, description="Objects skipped (boards), with reason (omitted if none)")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Objects that failed, with error (omitted if none)")


class MarkerInfo(BaseModel):
    count: int = Field(description="Number of markers placed on this object")
    ids: list[int] = Field(description="Contiguous ID block [lo, hi] (assembly-global)")


class MarkersResult(BaseModel):
    output_dir: str = Field(description="Type-named directory the marker JSON were written to")
    type: MarkerFamily = Field(description="Marker type used (aruco | apriltag)")
    dictionary: str = Field(description="Dictionary used (auto-selected to fit the assembly, or the override)")
    generated: list[str] = Field(description="Object names written (file = <name>_aruco.json)")
    markers: dict[str, MarkerInfo] = Field(description="Per object: {count, ids:[lo,hi]} — contiguous, assembly-global")
    skipped: Optional[list[SkipEntry]] = Field(default=None, description="Objects skipped, with reason (omitted if none)")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Objects that failed, with error (omitted if none)")
    warnings: Optional[list[str]] = Field(default=None, description="e.g. dictionary climbed past the robust default (omitted if none)")


class ExportResult(BaseModel):
    """One tool, both formats. Reads existing markers and renders PNG and/or PDF."""
    type: MarkerFamily = Field(description="Marker type rendered (aruco | apriltag)")
    formats: list[str] = Field(description="Formats exported: any of 'png', 'pdf'")
    output_dirs: dict[str, str] = Field(description="Per-format output directory {format: dir}")
    generated: list[str] = Field(description="Object names rendered (same set across formats)")
    skipped: Optional[list[SkipEntry]] = Field(default=None, description="Objects skipped, with reason (omitted if none)")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Per-format failures, error prefixed [png]/[pdf] (omitted if none)")


class GraspInfo(BaseModel):
    candidates: int = Field(description="Number of grasp candidates after filtering")
    axis: str = Field(description="Approach axis used")
    source_marker_id: int = Field(description="Marker the grasp was detected/transformed from")


class GraspResult(BaseModel):
    output_dir: str = Field(description="grasp_points directory")
    type: MarkerFamily = Field(description="Marker type the grasps were attached to")
    generated: list[str] = Field(description="Object names written (file = <name>_grasp_points.json)")
    grasps: dict[str, GraspInfo] = Field(description="Per object: {candidates, axis, source_marker_id}")
    skipped: Optional[list[SkipEntry]] = Field(default=None, description="Objects skipped (boards / no markers), with reason")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Objects that failed (e.g. no marker on the axis), with error")


def _split_results(results) -> tuple[list[str], Optional[list[SkipEntry]], Optional[list[FailEntry]]]:
    """Bucket stage results into the shared envelope's generated / skipped / failed lists."""
    generated = [r.object for r in results if r.status == "ok"]
    skipped = [SkipEntry(name=r.object or "?", reason=r.detail) for r in results if r.status == "skip"]
    failed = [FailEntry(name=r.object or "?", error=r.detail) for r in results if r.status == "fail"]
    return generated, (skipped or None), (failed or None)


@contextlib.contextmanager
def _quiet():
    """Redirect stage stdout to stderr. The CAD/PNG/PDF generators print verbosely; on a stdio
    MCP server stdout carries the JSON-RPC protocol, so any stray print corrupts it."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


# ── server ─────────────────────────────────────────────────────────────────────
mcp = FastMCP("aruco-pipeline")


@mcp.tool()
def generate_wireframe(
    assembly: Annotated[str, Field(description="Assembly recipe name, e.g. 'fmb1' (resolves assets/input/assemblies/<assembly>.json).")],
    object: Annotated[Optional[str], Field(description="Optional single CAD object name to narrow to; default = all objects in the assembly.")] = None,
) -> WireframeResult:
    """Extract the CAD wireframe (vertices + unique edges + bbox) for an assembly's objects.

    Deterministic and one-shot — regeneration overwrites in place. Writes
    assets/output/<assembly>/wireframe/<object>_wireframe.json per object.

    Returns:
        output_dir: the wireframe output directory (encodes the assembly)
        generated: object names written (file = <name>_wireframe.json)
        skipped: objects skipped with reason — omitted if none
        failed: objects that failed with error — omitted if none"""
    wf_dir = _output_dir(assembly) / "wireframe"
    components = resolve_components(assembly)
    if object is not None:
        components = [c for c in components if c.get("name") == object]
        if not components:
            return WireframeResult(
                output_dir=str(wf_dir), generated=[],
                failed=[FailEntry(name=object, error="object not in assembly")],
            ).model_dump(exclude_none=True)

    with _quiet():
        results = stage_wireframe(components, _input_dir() / "models", _output_dir(assembly), skip_existing=False)
    generated, skipped, failed = _split_results(results)
    return WireframeResult(
        output_dir=str(wf_dir),
        generated=generated, skipped=skipped, failed=failed,
    ).model_dump(exclude_none=True)


def _fold_summary(symmetry_file: Path) -> dict[str, int]:
    """Compact {x,y,z: N-fold} from a *_symmetry.json file."""
    fa = json.loads(symmetry_file.read_text()).get("fold_axes", {})
    return {ax: int(fa[ax]["fold"]) for ax in ("x", "y", "z") if ax in fa and "fold" in fa[ax]}


@mcp.tool()
def generate_symmetry(
    assembly: Annotated[str, Field(description="Assembly recipe name, e.g. 'fmb1' (resolves assets/input/assemblies/<assembly>.json).")],
    object: Annotated[Optional[str], Field(description="Optional single CAD object name to narrow to; default = all objects in the assembly.")] = None,
) -> SymmetryResult:
    """Auto-detect fold-symmetry axes for an assembly's objects (boards are skipped).

    Deterministic and one-shot — regeneration overwrites in place. Writes
    assets/output/<assembly>/symmetry/<object>_symmetry.json per non-board object.

    Returns:
        assembly: the assembly processed
        output_dir: the symmetry output directory
        generated: object names written (file = <name>_symmetry.json)
        folds: per object, N-fold per axis {x,y,z} (1 = none, 2 = 2-fold, 6 = hex)
        skipped: boards (and no-mesh objects), with reason — omitted if none
        failed: objects that failed, with error — omitted if none"""
    sym_dir = _output_dir(assembly) / "symmetry"
    components = resolve_components(assembly)
    if object is not None:
        components = [c for c in components if c.get("name") == object]
        if not components:
            return SymmetryResult(
                output_dir=str(sym_dir), generated=[], folds={},
                failed=[FailEntry(name=object, error="object not in assembly")],
            ).model_dump(exclude_none=True)

    with _quiet():
        results = stage_symmetry(components, _input_dir() / "models", _output_dir(assembly), skip_existing=False)
    generated, skipped, failed = _split_results(results)
    folds = {name: _fold_summary(sym_dir / f"{name}_symmetry.json") for name in generated}
    return SymmetryResult(
        output_dir=str(sym_dir),
        generated=generated, folds=folds, skipped=skipped, failed=failed,
    ).model_dump(exclude_none=True)


def _marker_info(aruco_file: Path) -> MarkerInfo:
    """Compact {count, ids:[lo,hi]} from a *_aruco.json file."""
    ids = [int(m["aruco_id"]) for m in json.loads(aruco_file.read_text()).get("markers", [])]
    return MarkerInfo(count=len(ids), ids=[min(ids), max(ids)] if ids else [])


@mcp.tool()
def generate_markers(
    assembly: Annotated[str, Field(description="Assembly recipe name, e.g. 'fmb1' (resolves assets/input/assemblies/<assembly>.json).")],
    object: Annotated[Optional[str], Field(description="Optional single CAD object to (re)generate; default = all objects. IDs are always computed over the full assembly.")] = None,
    type: Annotated[MarkerFamily, Field(description="Marker type: 'aruco' or 'apriltag'.")] = "aruco",
    dictionary: Annotated[Optional[str], Field(description="Optional dictionary override (same type, must fit the marker count); default = auto-select the smallest that fits.")] = None,
) -> MarkersResult:
    """Place fiducial markers on an assembly's objects with assembly-global contiguous IDs.

    IDs are allocated by assembly_order into contiguous per-object blocks and the dictionary is
    chosen from the whole-assembly total (adding objects climbs the ladder; the dictionary grows
    only when a capacity boundary is crossed). With `object`, the full numbering is still computed
    but only that object's file is written, at its correct block. Writes
    assets/output/<assembly>/<marker_family>/<object>_aruco.json.

    Returns:
        output_dir: the family-named marker directory
        marker_family / dictionary: family + dictionary actually used
        generated: object names written (file = <name>_aruco.json)
        markers: per object {count, ids:[lo,hi]} — contiguous, assembly-global
        skipped / failed: omitted if none
        warnings: e.g. dictionary climbed past the robust default — omitted if none"""
    family = type  # internal name; `type` is the agent-facing arg
    fam_dir = _output_dir(assembly) / family
    components = resolve_components(assembly)
    only: Optional[set[str]] = None
    if object is not None:
        if object not in {c["name"] for c in components}:
            return MarkersResult(
                output_dir=str(fam_dir), type=family, dictionary=dictionary or "",
                generated=[], markers={}, failed=[FailEntry(name=object, error="object not in assembly")],
            ).model_dump(exclude_none=True)
        only = {object}

    try:
        with _quiet():
            results, chosen, _total, warnings = stage_markers(
                components, _input_dir() / "models", _output_dir(assembly),
                family=family, border_width=None, only=only, dictionary=dictionary,
            )
    except ValueError as exc:  # bad dictionary override / capacity overflow
        return MarkersResult(
            output_dir=str(fam_dir), type=family, dictionary=dictionary or "",
            generated=[], markers={}, failed=[FailEntry(name=object or assembly, error=str(exc))],
        ).model_dump(exclude_none=True)

    generated, skipped, failed = _split_results(results)
    markers = {name: _marker_info(fam_dir / f"{name}_aruco.json") for name in generated}
    return MarkersResult(
        output_dir=str(fam_dir), type=family, dictionary=chosen or "",
        generated=generated, markers=markers, skipped=skipped, failed=failed,
        warnings=(warnings or None),
    ).model_dump(exclude_none=True)


def _render_one(asm_out: Path, family: str, kind: str, only: Optional[set[str]]):
    """Render one format from existing markers. kind = 'png' | 'pdf'. Returns (out_dir, results)."""
    out_dir = asm_out / f"{family}_{'pngs' if kind == 'png' else 'pdfs'}"
    stage = stage_marker_pngs if kind == "png" else stage_marker_pdfs
    with _quiet():
        results = stage(asm_out, family=family, skip_existing=False, only=only)
    return out_dir, results


@mcp.tool()
def export_fiducial(
    assembly: Annotated[str, Field(description="Assembly recipe name, e.g. 'fmb1'.")],
    object: Annotated[Optional[str], Field(description="Optional single object to export; default = all objects that have markers.")] = None,
    type: Annotated[MarkerFamily, Field(description="Marker type to render: 'aruco' or 'apriltag' (must already be generated).")] = "aruco",
    format: Annotated[ExportFormat, Field(description="'png' (detectable images), 'pdf' (A4 print-and-cut sheets), or 'both'.")] = "both",
) -> ExportResult:
    """Export existing markers as PNG and/or PDF — one tool for both (same backend, reads markers).

    PNG = one detectable square per marker (no cut border) under <type>_pngs/<object>/.
    PDF = the object's markers on an A4 print-and-cut sheet at true size under <type>_pdfs/.
    Run generate_markers first; this tool does not place markers.

    Returns:
        type / formats: marker type + which formats were exported
        output_dirs: {format: directory}
        generated: object names rendered (same set across formats)
        skipped / failed: omitted if none (failures prefixed [png]/[pdf])"""
    family = type
    asm_out = _output_dir(assembly)
    marker_dir = asm_out / family
    kinds = ["png", "pdf"] if format == "both" else [format]

    if not marker_dir.is_dir() or not any(marker_dir.glob("*_aruco.json")):
        return ExportResult(
            type=family, formats=[], output_dirs={}, generated=[],
            failed=[FailEntry(name=assembly, error=f"no {family} markers found — run generate_markers first")],
        ).model_dump(exclude_none=True)

    only: Optional[set[str]] = {object} if object is not None else None
    output_dirs: dict[str, str] = {}
    gen: set[str] = set()
    skip_by_name: dict[str, str] = {}
    failed: list[FailEntry] = []
    for kind in kinds:
        out_dir, results = _render_one(asm_out, family, kind, only)
        output_dirs[kind] = str(out_dir)
        if object is not None and not results:
            failed.append(FailEntry(name=object, error=f"[{kind}] no markers for that object (or not in assembly)"))
        for r in results:
            if r.status == "ok":
                gen.add(r.object or "?")
            elif r.status == "skip":
                skip_by_name[r.object or "?"] = r.detail
            else:
                failed.append(FailEntry(name=r.object or "?", error=f"[{kind}] {r.detail}"))

    skipped = [SkipEntry(name=n, reason=reason) for n, reason in skip_by_name.items()]
    return ExportResult(
        type=family, formats=kinds, output_dirs=output_dirs, generated=sorted(gen),
        skipped=(skipped or None), failed=(failed or None),
    ).model_dump(exclude_none=True)


def _load_gripper(gripper_axes: GripperAxes) -> dict:
    """assets/input/gripper.json -> backend filter_params. The file is REQUIRED (the caller
    guards its presence); missing individual keys fall back to standard values."""
    g = json.loads((_input_dir() / "gripper.json").read_text())
    return {
        "gripper_max_width_mm": float(g.get("max_width_mm", 100.0)),
        "grasp_clearance_mm": float(g.get("clearance_mm", 14.0)),
        "gripper_tip_thickness_mm": float(g.get("tip_thickness_mm", 20.0)),
        "check_x_axis": gripper_axes in ("x", "both"),
        "check_y_axis": gripper_axes in ("y", "both"),
    }


def _axis_marker_id(aruco_file: Path, axis: str) -> Optional[int]:
    """Marker id on the face the approach axis points along (None if that face has no marker)."""
    face = _AXIS_FACE[axis]
    for m in json.loads(aruco_file.read_text()).get("markers", []):
        if m.get("face_type") == face:
            return int(m["aruco_id"])
    return None


def _axis_thickness(wireframe_file: Path, axis: str) -> Optional[float]:
    """Object extent (m) along the approach axis, from the wireframe bbox (axis-aware thickness)."""
    bb = json.loads(wireframe_file.read_text()).get("mesh_info", {}).get("bounding_box")
    if not bb:
        return None
    i = _AXIS_INDEX[axis]
    return float(bb["max"][i]) - float(bb["min"][i])


def _grasp_info(grasp_file: Path, axis: str) -> GraspInfo:
    d = json.loads(grasp_file.read_text())
    return GraspInfo(candidates=int(d.get("total_grasp_points", 0)), axis=axis,
                     source_marker_id=int(d.get("source_marker_id", -1)))


@mcp.tool()
def generate_grasp_candidates(
    assembly: Annotated[str, Field(description="Assembly recipe name, e.g. 'fmb1'.")],
    object: Annotated[Optional[str], Field(description="Optional single object; default = all graspable (non-board) objects.")] = None,
    type: Annotated[MarkerFamily, Field(description="Which marker type's markers to attach grasps to: 'aruco' or 'apriltag'.")] = "aruco",
    axis: Annotated[GraspAxis, Field(description="Approach face for the top-down render: 'current' (= top +z) or +x/-x/+y/-y/+z/-z.")] = "current",
    gripper_axes: Annotated[GripperAxes, Field(description="Gripper CLOSING axes to check during filtering: 'x', 'y', or 'both'.")] = "both",
) -> GraspResult:
    """Detect + filter grasp candidates per object: top-down render from the approach `axis` ->
    2D candidate detection -> gripper-constraint filter -> transform to all markers.

    Gripper geometry comes from assets/input/gripper.json (defaults if absent). Z-thickness is
    auto-derived along the approach axis; camera distance is internal. Needs markers (`type`) and
    wireframe present (run generate_markers + generate_wireframe first). Boards are skipped.

    Returns:
        output_dir / type: grasp dir + marker type
        generated: object names written (file = <name>_grasp_points.json)
        grasps: per object {candidates, axis, source_marker_id}
        skipped: boards / no-markers, with reason — omitted if none
        failed: e.g. no marker on the requested axis — omitted if none"""
    family = type
    asm_out = _output_dir(assembly)
    grasp_dir = asm_out / "grasp_points"
    marker_dir = asm_out / family
    wf_dir = asm_out / "wireframe"

    if not marker_dir.is_dir() or not any(marker_dir.glob("*_aruco.json")):
        return GraspResult(output_dir=str(grasp_dir), type=family, generated=[], grasps={},
                           failed=[FailEntry(name=assembly, error=f"no {family} markers — run generate_markers first")]).model_dump(exclude_none=True)
    if not wf_dir.is_dir() or not any(wf_dir.glob("*_wireframe.json")):
        return GraspResult(output_dir=str(grasp_dir), type=family, generated=[], grasps={},
                           failed=[FailEntry(name=assembly, error="no wireframe — run generate_wireframe first")]).model_dump(exclude_none=True)
    gripper_file = _input_dir() / "gripper.json"
    if not gripper_file.exists():
        return GraspResult(output_dir=str(grasp_dir), type=family, generated=[], grasps={},
                           failed=[FailEntry(name=assembly, error="no gripper config — create assets/input/gripper.json (type, max_width_mm, clearance_mm, tip_thickness_mm)")]).model_dump(exclude_none=True)
    gripper_type = json.loads(gripper_file.read_text()).get("type", "parallel")
    if gripper_type != "parallel":
        return GraspResult(output_dir=str(grasp_dir), type=family, generated=[], grasps={},
                           failed=[FailEntry(name=assembly, error=f"gripper type {gripper_type!r} not supported — only 'parallel' is implemented")]).model_dump(exclude_none=True)

    components = resolve_components(assembly)
    if object is not None and object not in {c["name"] for c in components}:
        return GraspResult(output_dir=str(grasp_dir), type=family, generated=[], grasps={},
                           failed=[FailEntry(name=object, error="object not in assembly")]).model_dump(exclude_none=True)

    filter_params = _load_gripper(gripper_axes)
    resolvable: set[str] = set()
    marker_id_by_name: dict[str, int] = {}
    thickness_by_name: dict[str, float] = {}
    skipped: list[SkipEntry] = []
    failed: list[FailEntry] = []
    for comp in components:
        name = comp["name"]
        if object is not None and name != object:
            continue
        if comp.get("type") == "board":
            skipped.append(SkipEntry(name=name, reason="board"))
            continue
        aruco_f = marker_dir / f"{name}_aruco.json"
        if not aruco_f.exists():
            skipped.append(SkipEntry(name=name, reason="no markers"))
            continue
        mid = _axis_marker_id(aruco_f, axis)
        if mid is None:
            failed.append(FailEntry(name=name, error=f"no marker on the {axis} face"))
            continue
        marker_id_by_name[name] = mid
        th = _axis_thickness(wf_dir / f"{name}_wireframe.json", axis)
        if th is not None:
            thickness_by_name[name] = th
        resolvable.add(name)

    with _quiet():
        results = stage_grasp(
            components, _input_dir() / "models", asm_out, family=family, only=resolvable,
            marker_id_by_name=marker_id_by_name, thickness_by_name=thickness_by_name,
            filter_params=filter_params,
        )
    generated, st_skipped, st_failed = _split_results(results)
    skipped += st_skipped or []
    failed += st_failed or []
    grasps = {name: _grasp_info(grasp_dir / f"{name}_grasp_points.json", axis) for name in generated}
    return GraspResult(
        output_dir=str(grasp_dir), type=family, generated=generated, grasps=grasps,
        skipped=(skipped or None), failed=(failed or None),
    ).model_dump(exclude_none=True)


PIPELINE_STAGES = ["wireframe", "markers", "fiducial", "symmetry", "grasp"]
_PIPE_PREREQS = {"fiducial": ["markers"], "grasp": ["wireframe", "markers"]}


class AssemblySummary(BaseModel):
    wireframe: Optional[int] = Field(default=None, description="Wireframes written")
    markers: Optional[int] = Field(default=None, description="Marker sets written")
    dictionary: Optional[str] = Field(default=None, description="Dictionary chosen for this assembly")
    fiducial: Optional[int] = Field(default=None, description="Objects exported (png/pdf)")
    symmetry: Optional[int] = Field(default=None, description="Symmetry records written")
    grasp: Optional[int] = Field(default=None, description="Grasp candidate sets written")


class RunPipelineResult(BaseModel):
    type: MarkerFamily = Field(description="Marker type used across the run")
    stages: list[str] = Field(description="Stages actually run (prerequisites auto-included), in DAG order")
    assemblies: dict[str, AssemblySummary] = Field(description="Per assembly: per-stage generated counts (+ dictionary)")
    failed: Optional[list[FailEntry]] = Field(default=None, description="Aggregated failures, name = '<assembly>/<stage>/<object>' (omitted if none)")


def _call(tool, **kw):
    """Invoke another @mcp.tool() in-process (its plain function), returning the dict result."""
    return getattr(tool, "fn", tool)(**kw)


@mcp.tool()
def run_pipeline(
    assemblies: Annotated[list[str], Field(description="List of assembly recipe names (one or more), e.g. ['fmb1'] or ['fmb1','fmb2'].")],
    type: Annotated[MarkerFamily, Field(description="Marker type for the whole run: 'aruco' or 'apriltag'.")] = "aruco",
    axis: Annotated[GraspAxis, Field(description="Grasp approach face: 'current' (top +z) or +x/-x/+y/-y/+z/-z.")] = "current",
    gripper_axes: Annotated[GripperAxes, Field(description="Gripper closing axes for grasp filtering: 'x', 'y', or 'both'.")] = "both",
    stages: Annotated[Optional[list[str]], Field(description="Subset of ['wireframe','markers','fiducial','symmetry','grasp']; default = all. Prerequisites are auto-included.")] = None,
    format: Annotated[ExportFormat, Field(description="Fiducial export format: 'png', 'pdf', or 'both'.")] = "both",
) -> RunPipelineResult:
    """Run the full CAD->outputs pipeline for one or more assemblies — the catch-all that composes
    the five stage tools in DAG order (wireframe, markers, fiducial export, symmetry, grasp).

    Threads `type`/`axis`/`gripper_axes`/`format` to the relevant stages. A `stages` subset auto-
    includes prerequisites (e.g. 'grasp' pulls in 'wireframe'+'markers'). Gripper geometry still
    comes from assets/input/gripper.json (grasp errors if absent).

    Returns:
        type / stages: marker type + the stages actually run (post-prereq, DAG order)
        assemblies: per assembly {stage: generated_count, dictionary}
        failed: aggregated failures '<assembly>/<stage>/<object>' — omitted if none"""
    family = type
    asm_list = list(assemblies)
    req = set(stages) if stages else set(PIPELINE_STAGES)
    for dep, pre in _PIPE_PREREQS.items():
        if dep in req:
            req.update(pre)
    eff = [s for s in PIPELINE_STAGES if s in req]

    out: dict[str, AssemblySummary] = {}
    failed: list[FailEntry] = []

    def grab(a: str, stage: str, res: dict) -> int:
        for fe in res.get("failed") or []:
            failed.append(FailEntry(name=f"{a}/{stage}/{fe['name']}", error=fe["error"]))
        return len(res.get("generated", []))

    for a in asm_list:
        s = AssemblySummary()
        if "wireframe" in eff:
            s.wireframe = grab(a, "wireframe", _call(generate_wireframe, assembly=a))
        if "markers" in eff:
            r = _call(generate_markers, assembly=a, type=family)
            s.markers = grab(a, "markers", r)
            s.dictionary = r.get("dictionary")
        if "fiducial" in eff:
            s.fiducial = grab(a, "fiducial", _call(export_fiducial, assembly=a, type=family, format=format))
        if "symmetry" in eff:
            s.symmetry = grab(a, "symmetry", _call(generate_symmetry, assembly=a))
        if "grasp" in eff:
            s.grasp = grab(a, "grasp", _call(generate_grasp_candidates, assembly=a, type=family, axis=axis, gripper_axes=gripper_axes))
        out[a] = s

    return RunPipelineResult(
        type=family, stages=eff, assemblies=out, failed=(failed or None),
    ).model_dump(exclude_none=True)


if __name__ == "__main__":
    mcp.run(transport="stdio")
