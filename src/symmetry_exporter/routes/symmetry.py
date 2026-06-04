"""Symmetry data routes for Symmetry Exporter."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from shared.fastapi_utils import get_data_dir as _get_data_dir

from ..core.fold_axes import auto_detect_fold_axes
from .models import ExportSymmetryRequest

router = APIRouter(prefix="/api")

_MODEL_EXTS = (".obj", ".stl", ".ply")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


def _find_model(object_name: str) -> Path | None:
    """Resolve a CAD mesh for an object (first matching extension), or None."""
    models_dir = get_data_dir() / "models"
    for ext in _MODEL_EXTS:
        candidate = models_dir / f"{object_name}{ext}"
        if candidate.exists():
            return candidate
    return None


@router.get("/symmetry/{object_name}")
async def get_symmetry(object_name: str):
    """Get fold symmetry data for an object."""
    data_dir = get_data_dir()
    symmetry_file = data_dir / "symmetry" / f"{object_name}_symmetry.json"

    if not symmetry_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Symmetry data not found for {object_name}"
        )

    try:
        with open(symmetry_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading symmetry data: {str(e)}"
        )


@router.get("/symmetry/{object_name}/auto")
async def auto_detect_symmetry(object_name: str):
    """Auto-detect fold symmetry from CAD geometry (deterministic; no human input).

    This is the autoload that runs before human annotation: it computes fold_axes
    straight from the mesh. The existing POST /api/export-symmetry (human) overrides it.
    """
    mesh_path = _find_model(object_name)
    if mesh_path is None:
        raise HTTPException(
            status_code=404, detail=f"No CAD model found for {object_name}"
        )
    try:
        return auto_detect_fold_axes(mesh_path, object_name)
    except ValueError as e:
        # e.g. a continuous (revolution) symmetry the discrete schema can't represent
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Symmetry detection failed for {object_name}: {str(e)}"
        )


@router.post("/export-symmetry")
async def export_symmetry(data: ExportSymmetryRequest):
    """Export fold symmetry data to a JSON file in the data/symmetry folder."""
    object_name = data.object_name
    fold_axes = data.fold_axes

    if not object_name:
        raise HTTPException(status_code=400, detail="object_name is required")

    if not fold_axes:
        raise HTTPException(
            status_code=400, detail="fold_axes is required and cannot be empty"
        )

    data_dir = get_data_dir()
    symmetry_dir = data_dir / "symmetry"
    symmetry_dir.mkdir(exist_ok=True)

    export_data = {"object_name": object_name, "fold_axes": fold_axes}

    filename = f"{object_name}_symmetry.json"
    filepath = symmetry_dir / filename

    try:
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=4)

        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath.relative_to(data_dir.parent)),
            "message": "Symmetry data exported successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error exporting symmetry data: {str(e)}"
        )
