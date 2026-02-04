"""Marker conversion and transformation services."""

import warnings
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..models.marker import MarkerData


def marker_to_json(
    internal_id: int, marker: MarkerData, cad_object_info: dict
) -> dict[str, Any]:
    """Convert MarkerData to JSON format for frontend."""
    if cad_object_info is None:
        raise ValueError("No CAD model loaded. Please load a CAD model first.")

    cad_position = np.array(cad_object_info.get("position", [0.0, 0.0, 0.0]))
    cad_rotation = cad_object_info.get(
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

    cad_center_local = np.array(cad_object_info["center"])
    position_world = (
        cad_position
        + R_cad @ (np.array(marker.position) - cad_center_local)
        + R_cad @ cad_center_local
    )

    R_marker_to_object = marker.get_current_rotation_matrix()
    R_marker_to_world = R_cad @ R_marker_to_object

    rot_scipy = Rotation.from_matrix(R_marker_to_world)
    quat = rot_scipy.as_quat()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler = rot_scipy.as_euler("xyz")

    T_object_to_marker = marker.get_T_object_to_marker(cad_object_info["center"])

    return {
        "aruco_id": marker.aruco_id,
        "internal_id": internal_id,
        "dictionary": marker.dictionary,
        "face_type": marker.face_type,
        "surface_normal": marker.face_normal.tolist(),
        "size": marker.size,
        "border_width": marker.border_width,
        "in_plane_rotation_deg": marker.in_plane_rotation_deg,
        "translation_offset": {
            "x": float(marker.in_plane_translation[0]),
            "y": float(marker.in_plane_translation[1]),
        },
        "T_object_to_marker": T_object_to_marker,
        "pose_absolute": {
            "position": {
                "x": float(position_world[0]),
                "y": float(position_world[1]),
                "z": float(position_world[2]),
            },
            "rotation": {
                "roll": float(euler[0]),
                "pitch": float(euler[1]),
                "yaw": float(euler[2]),
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            },
        },
    }
