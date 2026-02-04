"""Symmetry data routes for Symmetry Exporter."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from shared.fastapi_utils import get_data_dir as _get_data_dir

router = APIRouter(prefix="/api")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


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


@router.post("/export-symmetry")
async def export_symmetry(data: dict[str, Any]):
    """Export fold symmetry data to a JSON file in the data/symmetry folder."""
    object_name = data.get("object_name")
    fold_axes = data.get("fold_axes", {})

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
