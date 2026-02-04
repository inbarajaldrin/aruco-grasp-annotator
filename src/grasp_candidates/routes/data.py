"""Data API routes for grasp candidates, wireframe, and ArUco data."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from shared.fastapi_utils import get_data_dir as _get_data_dir
from shared.data_loaders import (
    load_wireframe_data,
    load_aruco_data,
    load_grasp_data,
)

router = APIRouter(prefix="/api")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


def get_grasp_dir() -> Path:
    """Get the grasp data directory path."""
    return get_data_dir() / "grasp"


def get_grasp_candidates_dir() -> Path:
    """Get the grasp candidates directory path."""
    return get_data_dir() / "grasp_candidates"


@router.get("/objects")
async def get_objects() -> list[str]:
    """List available objects from grasp JSON files."""
    grasp_dir = get_grasp_dir()
    if not grasp_dir.exists():
        return []

    objects = [
        grasp_file.stem.replace("_grasp_points_all_markers", "")
        for grasp_file in grasp_dir.glob("*_grasp_points_all_markers.json")
    ]
    return sorted(objects)


@router.get("/grasp-data/{object_name}")
async def get_grasp_data(object_name: str):
    """Get grasp points data for an object (for positions only)."""
    data = load_grasp_data(get_data_dir(), object_name)
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"Grasp data not found for {object_name}"
        )
    return data


@router.get("/grasp-candidates/{object_name}")
async def get_grasp_candidates(object_name: str):
    """Get grasp candidates data for an object."""
    candidates_file = get_grasp_candidates_dir() / f"{object_name}_grasp_candidates.json"

    if not candidates_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Grasp candidates not found for {object_name}"
        )

    try:
        with open(candidates_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading grasp candidates: {str(e)}"
        )


@router.get("/aruco/{object_name}")
async def get_aruco_data_endpoint(object_name: str):
    """Get ArUco marker data for an object."""
    data = load_aruco_data(get_data_dir(), object_name)
    if data is None:
        return JSONResponse(
            content={"error": f"ArUco data not found for {object_name}"},
            status_code=404,
        )
    return data


@router.get("/wireframe/{object_name}")
async def get_wireframe(object_name: str):
    """Get wireframe data for object visualization."""
    data = load_wireframe_data(get_data_dir(), object_name)
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"Wireframe data not found for {object_name}"
        )
    return data
