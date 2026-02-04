"""Marker placement routes for ArUco Annotator."""

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..models.session import session_state
from ..services.mesh_service import determine_face_type, group_triangles_by_face

router = APIRouter(prefix="/api")


async def _add_marker_internal(config: dict[str, Any]):
    """Internal function to add a marker. Imports at call time to avoid circular imports."""
    from .markers import add_marker
    return await add_marker(config)


@router.get("/faces")
async def get_faces():
    """Get detected faces from the mesh."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    triangles = np.asarray(session_state.mesh.triangles)
    vertices = np.asarray(session_state.mesh.vertices)

    face_groups = group_triangles_by_face(triangles, vertices)

    faces = []
    for face_center, face_normal, face_area, triangle_indices in face_groups:
        face_type = determine_face_type(tuple(face_normal))
        faces.append(
            {
                "center": face_center.tolist(),
                "normal": face_normal.tolist(),
                "area": float(face_area),
                "triangle_count": len(triangle_indices),
                "face_type": face_type,
            }
        )

    return JSONResponse({"faces": faces})


@router.get("/faces/primary")
async def get_primary_faces():
    """Get the 6 primary faces."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    cad_info = session_state.cad_object_info
    bbox_min = np.array(session_state.mesh_info["bbox_min"])
    bbox_max = np.array(session_state.mesh_info["bbox_max"])
    center = np.array(cad_info["center"])

    faces = [
        {
            "name": "Left",
            "center": [bbox_min[0], center[1], center[2]],
            "normal": [-1.0, 0.0, 0.0],
        },
        {
            "name": "Right",
            "center": [bbox_max[0], center[1], center[2]],
            "normal": [1.0, 0.0, 0.0],
        },
        {
            "name": "Front",
            "center": [center[0], bbox_min[1], center[2]],
            "normal": [0.0, -1.0, 0.0],
        },
        {
            "name": "Back",
            "center": [center[0], bbox_max[1], center[2]],
            "normal": [0.0, 1.0, 0.0],
        },
        {
            "name": "Bottom",
            "center": [center[0], center[1], bbox_min[2]],
            "normal": [0.0, 0.0, -1.0],
        },
        {
            "name": "Top",
            "center": [center[0], center[1], bbox_max[2]],
            "normal": [0.0, 0.0, 1.0],
        },
    ]

    return JSONResponse({"faces": faces})


@router.post("/place-marker/random")
async def place_random_marker(config: dict[str, Any] = None):
    """Place marker at random face."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    triangles = np.asarray(session_state.mesh.triangles)
    vertices = np.asarray(session_state.mesh.vertices)

    face_groups = group_triangles_by_face(triangles, vertices)
    if len(face_groups) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")

    import random

    face_center, face_normal, face_area, triangle_indices = face_groups[
        random.randint(0, len(face_groups) - 1)
    ]

    if config is None:
        config = {}

    config.update(
        {
            "position": {
                "x": float(face_center[0]),
                "y": float(face_center[1]),
                "z": float(face_center[2]),
            },
            "normal": {
                "x": float(face_normal[0]),
                "y": float(face_normal[1]),
                "z": float(face_normal[2]),
            },
        }
    )

    return await _add_marker_internal(config)


@router.post("/place-marker/smart")
async def place_smart_marker(config: dict[str, Any] = None):
    """Place marker using smart auto placement."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    triangles = np.asarray(session_state.mesh.triangles)
    vertices = np.asarray(session_state.mesh.vertices)

    face_groups = group_triangles_by_face(triangles, vertices)
    if len(face_groups) == 0:
        raise HTTPException(status_code=400, detail="No faces detected")

    best_score = -1
    best_face = None

    for face_center, face_normal, face_area, triangle_indices in face_groups:
        score = 0
        if face_area > 0:
            score += min(50, face_area * 10000)
        up_alignment = abs(face_normal[2])
        score += up_alignment * 30
        score += min(20, len(triangle_indices) / 5)

        if score > best_score:
            best_score = score
            best_face = (face_center, face_normal, face_area, triangle_indices)

    if best_face is None:
        raise HTTPException(status_code=400, detail="Could not find suitable face")

    face_center, face_normal, face_area, triangle_indices = best_face

    if config is None:
        config = {}

    config.update(
        {
            "position": {
                "x": float(face_center[0]),
                "y": float(face_center[1]),
                "z": float(face_center[2]),
            },
            "normal": {
                "x": float(face_normal[0]),
                "y": float(face_normal[1]),
                "z": float(face_normal[2]),
            },
        }
    )

    return await _add_marker_internal(config)


@router.post("/place-marker/all-6")
async def place_all_6_faces(config: dict[str, Any] = None):
    """Place markers on all 6 primary faces."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if config is None:
        config = {}

    cad_info = session_state.cad_object_info
    bbox_min = np.array(session_state.mesh_info["bbox_min"])
    bbox_max = np.array(session_state.mesh_info["bbox_max"])
    center = np.array(cad_info["center"])

    faces = [
        {
            "center": [bbox_min[0], center[1], center[2]],
            "normal": [-1.0, 0.0, 0.0],
            "name": "Left",
        },
        {
            "center": [bbox_max[0], center[1], center[2]],
            "normal": [1.0, 0.0, 0.0],
            "name": "Right",
        },
        {
            "center": [center[0], bbox_min[1], center[2]],
            "normal": [0.0, -1.0, 0.0],
            "name": "Front",
        },
        {
            "center": [center[0], bbox_max[1], center[2]],
            "normal": [0.0, 1.0, 0.0],
            "name": "Back",
        },
        {
            "center": [center[0], center[1], bbox_min[2]],
            "normal": [0.0, 0.0, -1.0],
            "name": "Bottom",
        },
        {
            "center": [center[0], center[1], bbox_max[2]],
            "normal": [0.0, 0.0, 1.0],
            "name": "Top",
        },
    ]

    markers = []
    base_aruco_id = config.get("aruco_id", session_state.next_marker_id)

    for i, face in enumerate(faces):
        face_config = config.copy()
        face_config.update(
            {
                "aruco_id": base_aruco_id + i,
                "position": {
                    "x": face["center"][0],
                    "y": face["center"][1],
                    "z": face["center"][2],
                },
                "normal": {
                    "x": face["normal"][0],
                    "y": face["normal"][1],
                    "z": face["normal"][2],
                },
            }
        )
        result = await _add_marker_internal(face_config)
        markers.append(json.loads(result.body))

    return JSONResponse({"markers": markers})


@router.post("/place-marker/corner")
async def place_corner_markers(config: dict[str, Any] = None):
    """Place 4 markers on the corners of a selected primary face."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if config is None:
        config = {}

    cad_info = session_state.cad_object_info
    bbox_min = np.array(session_state.mesh_info["bbox_min"])
    bbox_max = np.array(session_state.mesh_info["bbox_max"])
    center = np.array(cad_info["center"])

    face_index = config.get("face_index")
    if face_index is None:
        raise HTTPException(status_code=400, detail="face_index is required")
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="face_index must be an integer")
    if face_index < 0 or face_index > 5:
        raise HTTPException(status_code=400, detail="Face index must be 0-5")

    face_definitions = [
        {
            "name": "Left",
            "normal": [-1.0, 0.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
            ],
        },
        {
            "name": "Right",
            "normal": [1.0, 0.0, 0.0],
            "corners": [
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
            ],
        },
        {
            "name": "Front",
            "normal": [0.0, -1.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
            ],
        },
        {
            "name": "Back",
            "normal": [0.0, 1.0, 0.0],
            "corners": [
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
            ],
        },
        {
            "name": "Bottom",
            "normal": [0.0, 0.0, -1.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
            ],
        },
        {
            "name": "Top",
            "normal": [0.0, 0.0, 1.0],
            "corners": [
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
            ],
        },
    ]

    selected_face = face_definitions[face_index]
    marker_size = config.get("size", 0.021)
    inset_distance = marker_size / 2.0

    inset_corners = []
    normal = np.array(selected_face["normal"])

    plane_axes = []
    for axis_idx in range(3):
        if abs(normal[axis_idx]) < 0.5:
            plane_axes.append(axis_idx)

    if len(plane_axes) != 2:
        raise HTTPException(
            status_code=400, detail="Invalid face normal - must have exactly 2 plane axes"
        )

    for corner_pos in selected_face["corners"]:
        corner = np.array(corner_pos)
        inset_corner = corner.copy()

        for axis_idx in plane_axes:
            if corner[axis_idx] == bbox_min[axis_idx]:
                inset_corner[axis_idx] = bbox_min[axis_idx] + inset_distance
            elif corner[axis_idx] == bbox_max[axis_idx]:
                inset_corner[axis_idx] = bbox_max[axis_idx] - inset_distance
            else:
                if corner[axis_idx] < center[axis_idx]:
                    inset_corner[axis_idx] = corner[axis_idx] + inset_distance
                else:
                    inset_corner[axis_idx] = corner[axis_idx] - inset_distance

        for axis_idx in range(3):
            if abs(normal[axis_idx]) > 0.5:
                if normal[axis_idx] < 0:
                    inset_corner[axis_idx] = bbox_min[axis_idx]
                else:
                    inset_corner[axis_idx] = bbox_max[axis_idx]

        for axis_idx in range(3):
            inset_corner[axis_idx] = np.clip(
                inset_corner[axis_idx], bbox_min[axis_idx], bbox_max[axis_idx]
            )

        inset_corners.append(inset_corner.tolist())

    markers = []
    base_aruco_id = config.get("aruco_id", session_state.next_marker_id)
    errors = []

    for i, corner_pos in enumerate(inset_corners):
        try:
            corner_config = config.copy()
            corner_config.update(
                {
                    "aruco_id": base_aruco_id + i,
                    "position": {
                        "x": corner_pos[0],
                        "y": corner_pos[1],
                        "z": corner_pos[2],
                    },
                    "normal": {
                        "x": selected_face["normal"][0],
                        "y": selected_face["normal"][1],
                        "z": selected_face["normal"][2],
                    },
                }
            )
            result = await _add_marker_internal(corner_config)
            marker_data = json.loads(result.body)
            markers.append(marker_data)
        except HTTPException as e:
            error_msg = f"Marker {i} (ArUco ID {base_aruco_id + i}): {e.detail}"
            errors.append(error_msg)
            continue
        except Exception as e:
            error_msg = f"Marker {i} at corner {corner_pos}: {str(e)}"
            errors.append(error_msg)
            continue

    return JSONResponse({"markers": markers, "warnings": errors if errors else None})
