"""Model loading and management routes for ArUco Annotator."""

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from scipy.spatial.transform import Rotation

from ..core.cad_loader import CADLoader
from ..models.session import session_state
from ..utils.aruco_utils import ArUcoGenerator

router = APIRouter(prefix="/api")

# Initialize components
cad_loader = CADLoader()


@router.post("/load-model")
async def load_model(file: UploadFile = File(...)):
    """Load a CAD model file."""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        mesh = cad_loader.load_file(tmp_path, input_units="auto")
        mesh_info = cad_loader.get_mesh_info(mesh)

        bbox_min = np.array(mesh_info["bbox_min"])
        bbox_max = np.array(mesh_info["bbox_max"])
        center = (bbox_min + bbox_max) / 2.0

        session_state.mesh = mesh
        session_state.mesh_info = mesh_info
        session_state.cad_object_info = {
            "center": center.tolist(),
            "dimensions": mesh_info["dimensions"],
            "position": [0.0, 0.0, 0.0],
            "rotation": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        session_state.current_file = file.filename
        session_state.markers = {}
        session_state.next_marker_id = 0

        vertices = np.asarray(mesh.vertices).flatten().tolist()
        normals = np.asarray(mesh.vertex_normals).flatten().tolist()
        faces = np.asarray(mesh.triangles).flatten().tolist()

        tmp_path.unlink()

        return JSONResponse(
            {
                "success": True,
                "vertices": vertices,
                "normals": normals,
                "faces": faces,
                "mesh_info": mesh_info,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get loaded model information."""
    if session_state.mesh is None:
        raise HTTPException(status_code=404, detail="No model loaded")

    return JSONResponse(
        {
            "mesh_info": session_state.mesh_info,
            "cad_object_info": session_state.cad_object_info,
        }
    )


@router.get("/config")
async def get_config():
    """Get current ArUco configuration."""
    return JSONResponse(
        {
            "dictionaries": ArUcoGenerator.get_available_dictionaries(),
            "default_dictionary": "DICT_4X4_50",
            "default_size": 0.021,
            "default_border_width": 0.05,
        }
    )


@router.get("/cad-pose")
async def get_cad_pose():
    """Get the current CAD object pose."""
    if session_state.cad_object_info is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded")

    cad_info = session_state.cad_object_info
    rotation = cad_info.get(
        "rotation",
        {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    )

    position = cad_info.get("position", cad_info.get("center", [0.0, 0.0, 0.0]))

    return JSONResponse(
        {
            "position": {
                "x": float(position[0])
                if isinstance(position, (list, tuple))
                else float(position.get("x", 0.0)),
                "y": float(position[1])
                if isinstance(position, (list, tuple))
                else float(position.get("y", 0.0)),
                "z": float(position[2])
                if isinstance(position, (list, tuple))
                else float(position.get("z", 0.0)),
            },
            "rotation": {
                "roll": float(rotation.get("roll", 0.0)),
                "pitch": float(rotation.get("pitch", 0.0)),
                "yaw": float(rotation.get("yaw", 0.0)),
                "quaternion": rotation.get(
                    "quaternion", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                ),
            },
        }
    )


@router.post("/cad-pose")
async def update_cad_pose(pose_data: dict):
    """Update the CAD object pose."""
    if session_state.cad_object_info is None:
        raise HTTPException(status_code=400, detail="No CAD model loaded")

    position_data = pose_data.get("position", {})
    rotation_data = pose_data.get("rotation", {})

    position = [
        float(position_data.get("x", 0.0)),
        float(position_data.get("y", 0.0)),
        float(position_data.get("z", 0.0)),
    ]

    roll = float(rotation_data.get("roll", 0.0))
    pitch = float(rotation_data.get("pitch", 0.0))
    yaw = float(rotation_data.get("yaw", 0.0))

    rot_scipy = Rotation.from_euler(
        "xyz", [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)]
    )
    quat = rot_scipy.as_quat()

    session_state.cad_object_info["position"] = position
    session_state.cad_object_info["rotation"] = {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }

    return JSONResponse({"success": True})
