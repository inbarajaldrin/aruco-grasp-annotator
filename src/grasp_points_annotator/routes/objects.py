"""Object and marker discovery routes for Grasp Points Annotator."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from shared.data_loaders import (
    get_available_objects as _get_available_objects,
    load_aruco_data,
)
from shared.fastapi_utils import get_data_dir

router = APIRouter(prefix="/api")

# Data directory path
DATA_DIR = get_data_dir(Path(__file__).parent.parent / "app.py")


def get_available_objects() -> list[str]:
    """Dynamically discover objects from models directory."""
    return _get_available_objects(DATA_DIR)


def get_markers_for_object(object_name: str) -> list[int]:
    """Get list of marker IDs for an object."""
    data = load_aruco_data(DATA_DIR, object_name)
    if data is None:
        return []
    marker_ids = [marker["aruco_id"] for marker in data.get("markers", [])]
    return sorted(marker_ids)


@router.get("/objects")
async def get_objects():
    """Get list of available objects."""
    try:
        objects = get_available_objects()
        print(f"Found {len(objects)} objects: {objects}")
        return {"objects": objects}
    except Exception as e:
        print(f"Error getting objects: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error discovering objects: {str(e)}")


@router.get("/markers/{object_name}")
async def get_markers(object_name: str):
    """Get markers for a specific object."""
    marker_ids = get_markers_for_object(object_name)
    return {"object_name": object_name, "markers": marker_ids}
