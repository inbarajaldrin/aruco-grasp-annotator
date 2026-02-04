"""Component loading routes for Assembly App."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from shared.fastapi_utils import get_data_dir as _get_data_dir

from ..services.aruco_transform import transform_aruco_data

router = APIRouter(prefix="/api")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


def get_available_components() -> list[str]:
    """Dynamically discover components from wireframe directory."""
    data_dir = get_data_dir()
    wireframe_dir = data_dir / "wireframe"
    if not wireframe_dir.exists():
        return []
    components = []
    for filepath in wireframe_dir.glob("*_wireframe.json"):
        component_name = filepath.stem.replace("_wireframe", "")
        components.append(component_name)
    return sorted(components)


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
                        print(f"Error transforming ArUco data for {component_name}: {e}")
                        import traceback

                        traceback.print_exc()
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
    available_components = get_available_components()
    if component_name not in available_components:
        raise HTTPException(status_code=404, detail="Component not found")

    try:
        wireframe_path = data_dir / "wireframe" / f"{component_name}_wireframe.json"
        aruco_path = data_dir / "aruco" / f"{component_name}_aruco.json"

        if not wireframe_path.exists():
            raise HTTPException(status_code=404, detail="Wireframe data not found")

        with open(wireframe_path, "r") as f:
            wireframe_data = json.load(f)

        aruco_data = None
        if aruco_path.exists():
            with open(aruco_path, "r") as f:
                aruco_raw = json.load(f)
            aruco_data = transform_aruco_data(aruco_raw)

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


@router.get("/grasp-points/{object_name}")
async def get_grasp_points(object_name: str):
    """Get grasp points data for an object from the data/grasp directory."""
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
