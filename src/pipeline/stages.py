"""
Per-stage wrappers: each runs ONE pipeline stage for ONE assembly, writing into that
assembly's output subfolder. Every stage calls the already-extracted, already-verified
core function in-process (no servers, no subprocesses) and writes with the exact same
serialization the interactive apps use, so outputs are drop-in compatible with the rest
of the toolchain.

Output layout for an assembly rooted at ``asm_out``::

    asm_out/wireframe/<obj>_wireframe.json          (indent=2)
    asm_out/<family>/<obj>_aruco.json               (indent=2; <family> = aruco|apriltag)
    asm_out/<family>_pngs/<obj>/<obj>_marker_NNN.png
    asm_out/<family>_pdfs/<obj>.pdf
    asm_out/symmetry/<obj>_symmetry.json            (indent=4; non-board only)
    asm_out/grasp_points/<obj>_grasp_points.json    (indent=2; non-board only)

Marker IDs are assembly-contiguous: objects are walked in (assembly_order, name) order and
each gets a block of ``id_base..`` IDs. The dictionary is chosen AFTER the whole assembly's
marker count is known (placement geometry is dictionary-independent), then stamped onto
every record — so the family/dictionary is pure metadata and never perturbs geometry.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .markers import dictionary_capacity, get_family, select_dictionary
from .spec import StageResult

_MODEL_EXTS = (".obj", ".stl", ".ply")


# --- shared helpers -------------------------------------------------------------------


def find_model(models_dir: Path, name: str) -> Path | None:
    for ext in _MODEL_EXTS:
        candidate = models_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_components(config_path: Path) -> list[dict[str, Any]]:
    """Parsed component dicts (name/type/subtype/assembly_order) from an assembly config."""
    data = json.loads(Path(config_path).read_text())
    return list(data.get("components", []))


def walk_order(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic assembly walk: by assembly_order, then name (ties + missing order)."""
    return sorted(
        components,
        key=lambda c: (c.get("assembly_order", 1_000_000), c.get("name", "")),
    )


def is_board(comp: dict[str, Any]) -> bool:
    return comp.get("type") == "board"


# --- wireframe ------------------------------------------------------------------------


def stage_wireframe(
    components: list[dict[str, Any]],
    models_dir: Path,
    asm_out: Path,
    *,
    skip_existing: bool = False,
) -> list[StageResult]:
    from aruco_annotator.core.cad_loader import CADLoader
    from aruco_annotator.core.wireframe import build_wireframe

    out_dir = asm_out / "wireframe"
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = CADLoader()
    results: list[StageResult] = []
    for comp in walk_order(components):
        name = comp["name"]
        target = out_dir / f"{name}_wireframe.json"
        if skip_existing and target.exists():
            results.append(StageResult(stage="wireframe", object=name, status="skip", detail="exists"))
            continue
        mesh_path = find_model(models_dir, name)
        if mesh_path is None:
            results.append(StageResult(stage="wireframe", object=name, status="skip", detail="no model"))
            continue
        try:
            mesh = loader.load_file(mesh_path, input_units="auto")
            record = build_wireframe(mesh)
            target.write_text(json.dumps(record, indent=2))
            results.append(StageResult(stage="wireframe", object=name, status="ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(StageResult(stage="wireframe", object=name, status="fail", detail=str(exc)))
    return results


# --- markers (aruco | apriltag) -------------------------------------------------------


def stage_markers(
    components: list[dict[str, Any]],
    models_dir: Path,
    asm_out: Path,
    *,
    family: str,
    border_width: float | None,
    skip_existing: bool = False,
    only: set[str] | None = None,
    dictionary: str | None = None,
) -> tuple[list[StageResult], str | None, int, list[str]]:
    """Place markers for the whole assembly with contiguous IDs + a chosen dictionary.

    IDs + dictionary are ALWAYS computed over the full assembly (so a part resolves to its
    real block). ``only`` restricts which files are written (option A: single-object regen
    keeps correct assembly-global IDs); results are reported for the written set. ``dictionary``
    overrides the auto-selection (validated to be a same-family dict that fits the total).

    Returns ``(results, chosen_dictionary, total_markers, warnings)``.
    """
    from aruco_annotator.core.aruco_placement import generate_aruco_for_object

    fam = get_family(family)
    border = border_width if border_width is not None else fam.default_border
    placeholder_dict = fam.ladder[0][0]  # geometry is dict-independent; restamped below

    out_dir = asm_out / family
    out_dir.mkdir(parents=True, exist_ok=True)

    # The markers stage is ASSEMBLY-ATOMIC under skip_existing: IDs are contiguous across the
    # whole assembly and the dictionary is chosen from the assembly total, so adding one object
    # can renumber IDs and change every record's dictionary. Partial skip is therefore unsound
    # -- we skip only when the ENTIRE set is already present, otherwise regenerate all of it
    # consistently (deterministic, so unchanged objects rewrite byte-identically modulo the
    # timestamp). Per-object skip would risk ID collisions + stale dictionaries.
    ordered = walk_order(components)
    targets = {c["name"]: out_dir / f"{c['name']}_aruco.json" for c in ordered}
    if skip_existing and all(t.exists() for t in targets.values()) and targets:
        total = 0
        chosen = None
        for name, t in targets.items():
            d = json.loads(t.read_text())
            total += int(d.get("total_markers", len(d.get("markers", []))))
            chosen = d.get("aruco_dictionary", chosen)
        results = [StageResult(stage="markers", object=n, status="skip", detail="exists") for n in targets]
        return results, chosen, total, []

    # Pass 1: place every object, accumulating id_base. Records held in memory.
    records: dict[str, dict[str, Any]] = {}
    results: list[StageResult] = []
    id_base = 0
    for comp in ordered:
        name = comp["name"]
        mesh_path = find_model(models_dir, name)
        if mesh_path is None:
            results.append(StageResult(stage="markers", object=name, status="skip", detail="no model"))
            continue
        try:
            record = generate_aruco_for_object(
                mesh_path,
                name,
                comp_type=comp.get("type", "object"),
                subtype=comp.get("subtype"),
                id_base=id_base,
                dictionary=placeholder_dict,
                border_width=border,
            )
            records[name] = record
            id_base += len(record.get("markers", []))
            results.append(StageResult(stage="markers", object=name, status="ok"))
        except NotImplementedError as exc:
            results.append(StageResult(stage="markers", object=name, status="skip", detail=str(exc)))
        except Exception as exc:  # noqa: BLE001
            results.append(StageResult(stage="markers", object=name, status="fail", detail=str(exc)))

    total = id_base
    # Pass 2: pick the dictionary (override or auto), stamp + write the requested subset.
    if dictionary is not None:
        cap = dictionary_capacity(family, dictionary)
        if cap is None:
            raise ValueError(f"{dictionary!r} is not a {family} dictionary")
        if total > cap:
            raise ValueError(f"{dictionary} holds {cap} IDs < {total} markers needed")
        chosen, warnings = dictionary, []
    else:
        chosen, warnings = select_dictionary(family, total)
    for name, record in records.items():
        record["aruco_dictionary"] = chosen
        if only is None or name in only:
            (out_dir / f"{name}_aruco.json").write_text(json.dumps(record, indent=2))

    if only is not None:
        results = [r for r in results if r.object in only]
    return results, chosen, total, warnings


# --- marker PNGs / PDFs (render from the marker JSON; family-agnostic via cv2.aruco) ---


def stage_marker_pngs(asm_out: Path, *, family: str, skip_existing: bool = False, only: set[str] | None = None) -> list[StageResult]:
    import cv2
    from aruco_annotator.scripts.generate_aruco_png import ArUcoPNGGenerator

    json_dir = asm_out / family
    out_root = asm_out / f"{family}_pngs"
    results: list[StageResult] = []
    for json_file in sorted(json_dir.glob("*_aruco.json")):
        name = json_file.name[: -len("_aruco.json")]
        if only is not None and name not in only:
            continue
        out_dir = out_root / name
        if skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            results.append(StageResult(stage="markers_png", object=name, status="skip", detail="exists"))
            continue
        try:
            data = json.loads(json_file.read_text())
            dict_name = data.get("aruco_dictionary", "DICT_4X4_50")
            gen = ArUcoPNGGenerator(
                marker_size_mm=float(data["size"]) * 1000.0,
                border_width_percent=float(data["border_width"]) * 100.0,
                dpi=304.8,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            for marker in data["markers"]:
                mid = int(marker["aruco_id"])
                img = gen.generate_marker_image(mid, dict_name, add_black_border=False)
                cv2.imwrite(str(out_dir / f"{name}_marker_{mid:03d}.png"), img)
            results.append(StageResult(stage="markers_png", object=name, status="ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(StageResult(stage="markers_png", object=name, status="fail", detail=str(exc)))
    return results


def stage_marker_pdfs(asm_out: Path, *, family: str, skip_existing: bool = False, only: set[str] | None = None) -> list[StageResult]:
    from aruco_annotator.scripts.generate_aruco_pdf import ArUcoPDFGenerator

    json_dir = asm_out / family
    out_dir = asm_out / f"{family}_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[StageResult] = []
    for json_file in sorted(json_dir.glob("*_aruco.json")):
        name = json_file.name[: -len("_aruco.json")]
        if only is not None and name not in only:
            continue
        target = out_dir / f"{name}.pdf"
        if skip_existing and target.exists():
            results.append(StageResult(stage="markers_pdf", object=name, status="skip", detail="exists"))
            continue
        try:
            data = json.loads(json_file.read_text())
            ids = [int(m["aruco_id"]) for m in data["markers"]]
            gen = ArUcoPDFGenerator(
                marker_size_mm=float(data["size"]) * 1000.0,
                border_width_percent=float(data["border_width"]) * 100.0,
                page_size="A4",
                margin_mm=20,
                markers_per_row=3,
            )
            gen.generate_pdf(
                output_path=str(target),
                marker_ids=ids,
                dictionary=data.get("aruco_dictionary", "DICT_4X4_50"),
                add_black_border=True,
                black_border_mm=2.0,
            )
            results.append(StageResult(stage="markers_pdf", object=name, status="ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(StageResult(stage="markers_pdf", object=name, status="fail", detail=str(exc)))
    return results


# --- symmetry (non-board only) --------------------------------------------------------


def stage_symmetry(
    components: list[dict[str, Any]],
    models_dir: Path,
    asm_out: Path,
    *,
    skip_existing: bool = False,
) -> list[StageResult]:
    from symmetry_exporter.core.fold_axes import auto_detect_fold_axes

    out_dir = asm_out / "symmetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[StageResult] = []
    for comp in walk_order(components):
        name = comp["name"]
        if is_board(comp):
            results.append(StageResult(stage="symmetry", object=name, status="skip", detail="board"))
            continue
        target = out_dir / f"{name}_symmetry.json"
        if skip_existing and target.exists():
            results.append(StageResult(stage="symmetry", object=name, status="skip", detail="exists"))
            continue
        mesh_path = find_model(models_dir, name)
        if mesh_path is None:
            results.append(StageResult(stage="symmetry", object=name, status="skip", detail="no model"))
            continue
        try:
            record = auto_detect_fold_axes(mesh_path, name)
            target.write_text(json.dumps(record, indent=4))
            results.append(StageResult(stage="symmetry", object=name, status="ok"))
        except Exception as exc:  # noqa: BLE001
            results.append(StageResult(stage="symmetry", object=name, status="fail", detail=str(exc)))
    return results


# --- grasp (non-board; reads freshly-produced wireframe + markers from asm_out) --------


def stage_grasp(
    components: list[dict[str, Any]],
    models_dir: Path,
    asm_out: Path,
    *,
    family: str,
    skip_existing: bool = False,
    only: set[str] | None = None,
    marker_id_by_name: dict[str, int] | None = None,
    thickness_by_name: dict[str, float] | None = None,
    filter_params: dict[str, Any] | None = None,
) -> list[StageResult]:
    """Detect + filter grasp candidates per non-board object.

    ``marker_id_by_name`` picks the source marker per object (the approach axis, resolved by the
    caller); ``thickness_by_name`` overrides the auto Z-thickness (axis-aware); ``filter_params``
    carries the gripper config + which closing axes to check. All default to the backend's own
    behaviour when omitted. ``only`` restricts which objects are processed.
    """
    from grasp_points_annotator.core.grasp_pipeline import generate_grasp_points

    out_dir = asm_out / "grasp_points"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose a data_dir the grasp pipeline understands (fixed subdir names): models from
    # the input, markers (the chosen family) + wireframe from this assembly's outputs, and
    # the final grasp json lands back in asm_out/grasp_points. Both temp dirs are removed
    # in the finally block so repeated runs don't leak /tmp/grasp_* trees.
    staging = Path(tempfile.mkdtemp(prefix="grasp_stage_"))
    work = Path(tempfile.mkdtemp(prefix="grasp_work_"))
    results: list[StageResult] = []
    try:
        (staging / "models").symlink_to(models_dir.resolve())
        (staging / "aruco").symlink_to((asm_out / family).resolve())
        (staging / "wireframe").symlink_to((asm_out / "wireframe").resolve())
        (staging / "grasp_points").symlink_to(out_dir.resolve())

        for comp in walk_order(components):
            name = comp["name"]
            if only is not None and name not in only:
                continue
            if is_board(comp):
                results.append(StageResult(stage="grasp", object=name, status="skip", detail="board"))
                continue
            target = out_dir / f"{name}_grasp_points.json"
            if skip_existing and target.exists():
                results.append(StageResult(stage="grasp", object=name, status="skip", detail="exists"))
                continue
            if not (asm_out / family / f"{name}_aruco.json").exists():
                results.append(StageResult(stage="grasp", object=name, status="skip", detail="no markers"))
                continue
            try:
                generate_grasp_points(
                    name, data_dir=staging, outputs_dir=work,
                    marker_id=(marker_id_by_name or {}).get(name),
                    object_thickness=(thickness_by_name or {}).get(name),
                    filter_params=filter_params,
                )
                results.append(StageResult(stage="grasp", object=name, status="ok"))
            except Exception as exc:  # noqa: BLE001
                results.append(StageResult(stage="grasp", object=name, status="fail", detail=str(exc)))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    return results
