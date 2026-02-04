"""Marker management routes for ArUco Annotator."""

import json
import warnings
from typing import Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from scipy.spatial.transform import Rotation

from ..models.marker import MarkerData
from ..models.session import session_state
from ..services.marker_service import marker_to_json
from ..services.mesh_service import determine_face_type, group_triangles_by_face

router = APIRouter(prefix="/api")


@router.post("/add-marker")
async def add_marker(config: dict[str, Any]):
    """Add an ArUco marker at specified position."""
    if session_state.mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        dictionary = config.get("dictionary", "DICT_4X4_50")
        aruco_marker_id = config.get("aruco_id", session_state.next_marker_id)
        size = config.get("size", 0.021)
        border_width = config.get("border_width", 0.05)

        pos_data = config.get("position", {})
        normal_data = config.get("normal", {"x": 0, "y": 0, "z": 1})

        position_world = np.array(
            [pos_data.get("x", 0), pos_data.get("y", 0), pos_data.get("z", 0)]
        )
        normal_world = np.array(
            [normal_data.get("x", 0), normal_data.get("y", 0), normal_data.get("z", 1)]
        )

        cad_info = session_state.cad_object_info
        cad_position = np.array(cad_info.get("position", [0.0, 0.0, 0.0]))
        cad_rotation = cad_info.get(
            "rotation",
            {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

        if "quaternion" in cad_rotation:
            quat = cad_rotation["quaternion"]
            R_cad = Rotation.from_quat(
                [quat["x"], quat["y"], quat["z"], quat["w"]]
            ).as_matrix()
        else:
            R_cad = Rotation.from_euler(
                "xyz",
                [
                    np.deg2rad(cad_rotation.get("roll", 0.0)),
                    np.deg2rad(cad_rotation.get("pitch", 0.0)),
                    np.deg2rad(cad_rotation.get("yaw", 0.0)),
                ],
            ).as_matrix()

        position_local = R_cad.T @ (position_world - cad_position)
        normal_local = R_cad.T @ normal_world

        for existing_id, existing_marker in session_state.markers.items():
            if (
                existing_marker.dictionary == dictionary
                and existing_marker.aruco_id == aruco_marker_id
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"ArUco marker {dictionary} ID:{aruco_marker_id} already exists",
                )

        face_type = determine_face_type(tuple(normal_local))

        marker = MarkerData(
            aruco_id=aruco_marker_id,
            dictionary=dictionary,
            size=size,
            border_width=border_width,
            position=tuple(position_local),
            face_normal=tuple(normal_local),
            face_type=face_type,
        )

        internal_id = session_state.next_marker_id
        session_state.markers[internal_id] = marker
        session_state.next_marker_id += 1

        return JSONResponse(
            marker_to_json(internal_id, marker, session_state.cad_object_info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markers")
async def get_markers():
    """Get all markers."""
    if session_state.cad_object_info is None:
        raise HTTPException(
            status_code=400,
            detail="No CAD model loaded. Please load a CAD model first.",
        )

    markers_list = []
    for internal_id, marker in session_state.markers.items():
        markers_list.append(
            marker_to_json(internal_id, marker, session_state.cad_object_info)
        )

    return JSONResponse({"markers": markers_list})


@router.delete("/markers/{marker_id}")
async def delete_marker(marker_id: int):
    """Delete a marker."""
    if marker_id not in session_state.markers:
        raise HTTPException(status_code=404, detail="Marker not found")

    del session_state.markers[marker_id]
    return JSONResponse({"success": True})


@router.delete("/markers")
async def clear_all_markers():
    """Clear all markers."""
    session_state.markers = {}
    session_state.next_marker_id = 0
    return JSONResponse({"success": True})


@router.patch("/markers/{marker_id}/rotation")
async def update_marker_rotation(
    marker_id: int, rotation: dict[str, Any] = Body(...)
):
    """Update marker in-plane rotation."""
    marker_id = int(marker_id)

    if marker_id not in session_state.markers:
        available_ids = list(session_state.markers.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Marker not found. Requested ID: {marker_id}, Available IDs: {available_ids}",
        )

    marker = session_state.markers[marker_id]
    mode = rotation.get("mode", "relative")

    if mode == "absolute":
        yaw_deg = float(rotation.get("yaw", 0))
        marker.set_in_plane_rotation(yaw_deg)
    else:
        yaw_delta_deg = float(rotation.get("yaw", 0))
        new_rotation_deg = marker.in_plane_rotation_deg + yaw_delta_deg
        marker.set_in_plane_rotation(new_rotation_deg)

    R_current = marker.get_current_rotation_matrix()
    rot_scipy = Rotation.from_matrix(R_current)
    quat = rot_scipy.as_quat()

    axis_angle = rot_scipy.as_rotvec()
    angle_rad = np.linalg.norm(axis_angle)
    if angle_rad > 1e-6:
        axis = axis_angle / angle_rad
    else:
        axis = np.array([0, 0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = rot_scipy.as_euler("xyz")

    return JSONResponse(
        {
            "success": True,
            "in_plane_rotation_deg": float(marker.in_plane_rotation_deg),
            "rotation": [float(e) for e in euler],
            "quaternion": {
                "x": float(quat[0]),
                "y": float(quat[1]),
                "z": float(quat[2]),
                "w": float(quat[3]),
            },
            "axis_angle": {
                "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
                "angle_rad": float(angle_rad),
                "angle_deg": float(np.degrees(angle_rad)),
            },
        }
    )


@router.patch("/markers/{marker_id}/translation")
async def update_marker_translation(
    marker_id: int, translation: dict[str, Any] = Body(...)
):
    """Update marker position by translating in-plane."""
    if marker_id not in session_state.markers:
        raise HTTPException(status_code=404, detail="Marker not found")

    marker = session_state.markers[marker_id]

    mode = translation.get("mode", "relative")
    x_delta = float(translation.get("x", 0.0))
    y_delta = float(translation.get("y", 0.0))

    R_base = marker.base_rotation_matrix
    marker_x_axis = R_base[:, 0]
    marker_y_axis = R_base[:, 1]

    if mode == "absolute":
        marker.in_plane_translation = np.array([x_delta, y_delta])
    else:
        marker.in_plane_translation = marker.in_plane_translation + np.array(
            [x_delta, y_delta]
        )

    translation_vector = (
        marker.in_plane_translation[0] * marker_x_axis
        + marker.in_plane_translation[1] * marker_y_axis
    )

    if not hasattr(marker, "initial_position") or marker.initial_position is None:
        if np.linalg.norm(marker.in_plane_translation) > 1e-6:
            prev_translation = (
                marker.in_plane_translation[0] * marker_x_axis
                + marker.in_plane_translation[1] * marker_y_axis
            )
            marker.initial_position = marker.position - prev_translation
        else:
            marker.initial_position = marker.position.copy()

    marker.position = marker.initial_position + translation_vector

    return JSONResponse(
        {
            "success": True,
            "position": {
                "x": float(marker.position[0]),
                "y": float(marker.position[1]),
                "z": float(marker.position[2]),
            },
            "translation_offset": {
                "x": float(marker.in_plane_translation[0]),
                "y": float(marker.in_plane_translation[1]),
            },
        }
    )


@router.post("/markers/swap")
async def swap_marker_positions(swap_data: dict[str, Any]):
    """Swap ArUco IDs between two markers."""
    marker1_id = swap_data.get("marker1_id")
    marker2_id = swap_data.get("marker2_id")

    if marker1_id is None:
        raise HTTPException(status_code=400, detail="marker1_id is missing")
    if marker2_id is None:
        raise HTTPException(status_code=400, detail="marker2_id is missing")

    try:
        marker1_id = int(marker1_id)
        marker2_id = int(marker2_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Marker IDs must be integers")

    if marker1_id not in session_state.markers:
        raise HTTPException(status_code=404, detail=f"Marker {marker1_id} not found")
    if marker2_id not in session_state.markers:
        raise HTTPException(status_code=404, detail=f"Marker {marker2_id} not found")

    if marker1_id == marker2_id:
        raise HTTPException(status_code=400, detail="Cannot swap marker with itself")

    marker1 = session_state.markers[marker1_id]
    marker2 = session_state.markers[marker2_id]

    aruco_id1 = marker1.aruco_id
    aruco_id2 = marker2.aruco_id

    if aruco_id1 == aruco_id2:
        raise HTTPException(
            status_code=400, detail="Markers already have the same ArUco ID"
        )

    marker1.aruco_id = aruco_id2
    marker2.aruco_id = aruco_id1

    return JSONResponse({"success": True, "swapped": [marker1_id, marker2_id]})
