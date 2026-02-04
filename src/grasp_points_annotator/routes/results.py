"""Results and download routes for Grasp Points Annotator."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from shared.fastapi_utils import get_data_dir

router = APIRouter(prefix="/api")

# Data directory path
DATA_DIR = get_data_dir(Path(__file__).parent.parent / "app.py")


@router.get("/results/{object_name}")
async def get_results(object_name: str):
    """Get results for an object."""
    result_file = DATA_DIR / "grasp" / f"{object_name}_grasp_points_all_markers.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    with open(result_file, 'r') as f:
        data = json.load(f)

    return data


@router.get("/download/{object_name}")
async def download_results(object_name: str):
    """Download the final results JSON file."""
    result_file = DATA_DIR / "grasp_points" / f"{object_name}_grasp_points.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    return FileResponse(
        path=str(result_file),
        filename=f"{object_name}_grasp_points.json",
        media_type="application/json"
    )
