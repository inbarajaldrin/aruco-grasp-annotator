"""Component loading routes for Symmetry Exporter."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from shared.fastapi_utils import get_data_dir as _get_data_dir
from shared.data_loaders import (
    get_available_components as _get_available_components,
    transform_aruco_data,
)

router = APIRouter(prefix="/api")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


def get_available_components() -> list[str]:
    """Dynamically discover components from wireframe directory."""
    return _get_available_components(get_data_dir())


@router.get("/components")
async def get_components():
    """Get all available components with their wireframe and ArUco data."""
    data_dir = get_data_dir()
    components = {}

    available_components = get_available_components()

    for component_name in available_components:
        try:
            wireframe_path = data_dir / "wireframe" / f"{component_name}_wireframe.json"
            aruco_path = data_dir / "aruco" / f"{component_name}_aruco.json"

            if wireframe_path.exists():
                with open(wireframe_path, "r") as f:
                    wireframe_data = json.load(f)

                aruco_data = None
                if aruco_path.exists():
                    try:
                        with open(aruco_path, "r") as f:
                            aruco_raw = json.load(f)
                        aruco_data = transform_aruco_data(aruco_raw)
                    except Exception as e:
                        print(
                            f"Warning: Error transforming ArUco data for {component_name}: {e}"
                        )
                        aruco_data = None

                components[component_name] = {
                    "wireframe": wireframe_data,
                    "aruco": aruco_data,
                    "name": component_name,
                    "display_name": component_name.replace("_scaled70", "")
                    .replace("_", " ")
                    .title(),
                }
            else:
                print(f"Warning: Wireframe file not found for {component_name}")

        except Exception as e:
            print(f"Error loading component {component_name}: {e}")
            continue

    return components


@router.get("/components/{component_name}")
async def get_component(component_name: str):
    """Get a specific component's data."""
    data_dir = get_data_dir()
    wireframe_path = data_dir / "wireframe" / f"{component_name}_wireframe.json"
    if not wireframe_path.exists():
        raise HTTPException(status_code=404, detail="Component not found")

    try:
        aruco_path = data_dir / "aruco" / f"{component_name}_aruco.json"

        with open(wireframe_path, "r") as f:
            wireframe_data = json.load(f)

        aruco_data = None
        if aruco_path.exists():
            try:
                with open(aruco_path, "r") as f:
                    aruco_raw = json.load(f)
                aruco_data = transform_aruco_data(aruco_raw)
            except Exception as e:
                print(
                    f"Warning: Error transforming ArUco data for {component_name}: {e}"
                )
                aruco_data = None

        return {
            "wireframe": wireframe_data,
            "aruco": aruco_data,
            "name": component_name,
            "display_name": component_name.replace("_scaled70", "")
            .replace("_", " ")
            .title(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/grasp-data/{object_name}")
async def get_grasp_data(object_name: str):
    """Get grasp points data for an object."""
    data_dir = get_data_dir()
    grasp_file = data_dir / "grasp" / f"{object_name}_grasp_points_all_markers.json"

    if not grasp_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Grasp data not found for {object_name}"
        )

    try:
        with open(grasp_file, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading grasp data: {str(e)}"
        )
