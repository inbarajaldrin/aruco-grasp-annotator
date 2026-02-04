"""
Data loading utilities shared across applications.

Provides functions for discovering and loading data files including
wireframes, ArUco markers, and grasp points.
"""

import json
from pathlib import Path
from typing import Any, Optional


def get_available_components(data_dir: Path) -> list[str]:
    """
    Discover available components from the wireframe directory.

    Components are identified by files matching *_wireframe.json pattern.

    Args:
        data_dir: Path to the data directory

    Returns:
        Sorted list of component names
    """
    wireframe_dir = data_dir / "wireframe"
    components = []

    if wireframe_dir.exists():
        for wireframe_file in wireframe_dir.glob("*_wireframe.json"):
            component_name = wireframe_file.stem.replace("_wireframe", "")
            components.append(component_name)

    return sorted(components)


def get_available_objects(data_dir: Path) -> list[str]:
    """
    Discover available objects that have both CAD models and ArUco annotations.

    Objects are identified by matching ArUco annotation files with CAD models.

    Args:
        data_dir: Path to the data directory

    Returns:
        Sorted list of object names
    """
    models_dir = data_dir / "models"
    aruco_dir = data_dir / "aruco"
    objects = []

    if not models_dir.exists() or not aruco_dir.exists():
        return []

    aruco_files = list(aruco_dir.glob("*_aruco.json"))

    for aruco_file in aruco_files:
        object_name = aruco_file.stem.replace("_aruco", "")

        for ext in [".obj", ".stl", ".ply"]:
            model_file = models_dir / f"{object_name}{ext}"
            if model_file.exists():
                objects.append(object_name)
                break

    return sorted(objects)


def load_wireframe_data(data_dir: Path, component_name: str) -> Optional[dict[str, Any]]:
    """
    Load wireframe data for a component.

    Args:
        data_dir: Path to the data directory
        component_name: Name of the component

    Returns:
        Wireframe data dictionary or None if not found
    """
    wireframe_file = data_dir / "wireframe" / f"{component_name}_wireframe.json"

    if not wireframe_file.exists():
        return None

    with open(wireframe_file, "r") as f:
        return json.load(f)


def load_aruco_data(data_dir: Path, object_name: str) -> Optional[dict[str, Any]]:
    """
    Load ArUco marker data for an object.

    Args:
        data_dir: Path to the data directory
        object_name: Name of the object

    Returns:
        ArUco data dictionary or None if not found
    """
    aruco_file = data_dir / "aruco" / f"{object_name}_aruco.json"

    if not aruco_file.exists():
        return None

    with open(aruco_file, "r") as f:
        return json.load(f)


def load_grasp_data(data_dir: Path, object_name: str) -> Optional[dict[str, Any]]:
    """
    Load grasp points data for an object.

    Args:
        data_dir: Path to the data directory
        object_name: Name of the object

    Returns:
        Grasp data dictionary or None if not found
    """
    grasp_file = data_dir / "grasp" / f"{object_name}_grasp_points_all_markers.json"

    if not grasp_file.exists():
        return None

    with open(grasp_file, "r") as f:
        return json.load(f)


def transform_aruco_data(aruco_raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Transform ArUco data from T_object_to_marker format to pose_absolute format.

    This function handles both old and new ArUco data formats, converting
    T_object_to_marker to pose_absolute for frontend compatibility.

    Args:
        aruco_raw: Raw ArUco data dictionary

    Returns:
        Transformed ArUco data dictionary or None if input is None

    Raises:
        ValueError: If the ArUco data format is invalid
    """
    if not aruco_raw:
        return None

    if "markers" not in aruco_raw or not isinstance(aruco_raw["markers"], list):
        raise ValueError(
            f"ArUco data must have a 'markers' array. Got keys: {list(aruco_raw.keys())}"
        )

    if len(aruco_raw["markers"]) == 0:
        raise ValueError("ArUco data has empty 'markers' array")

    transformed_markers = []
    default_size = aruco_raw.get("size", 0.021)

    for idx, marker in enumerate(aruco_raw["markers"]):
        if not isinstance(marker, dict):
            raise ValueError(
                f"Marker at index {idx} is not a dictionary: {type(marker)}"
            )

        # Already in transformed format
        if "pose_absolute" in marker:
            transformed_marker = marker.copy()
            if "size" not in transformed_marker:
                transformed_marker["size"] = marker.get("size", default_size)
            transformed_markers.append(transformed_marker)
            continue

        # Need to transform from T_object_to_marker format
        if "T_object_to_marker" not in marker:
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} missing "
                f"'T_object_to_marker' or 'pose_absolute'. Marker keys: {list(marker.keys())}"
            )

        if not isinstance(marker["T_object_to_marker"], dict):
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} has invalid "
                f"'T_object_to_marker' type: {type(marker['T_object_to_marker'])}"
            )

        t_obj_to_marker = marker["T_object_to_marker"]
        if "position" not in t_obj_to_marker or "rotation" not in t_obj_to_marker:
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} T_object_to_marker "
                f"missing 'position' or 'rotation'. Keys: {list(t_obj_to_marker.keys())}"
            )

        transformed_marker = {
            "aruco_id": marker.get("aruco_id"),
            "size": marker.get("size", default_size),
            "face_type": marker.get("face_type"),
            "surface_normal": marker.get("surface_normal"),
            "pose_absolute": {
                "position": t_obj_to_marker.get("position", {"x": 0, "y": 0, "z": 0}),
                "rotation": t_obj_to_marker.get(
                    "rotation", {"roll": 0, "pitch": 0, "yaw": 0}
                ),
            },
        }

        transformed_markers.append(transformed_marker)

    return {
        "markers": transformed_markers,
        "aruco_dictionary": aruco_raw.get("aruco_dictionary", "DICT_4X4_50"),
        "size": default_size,
        "border_width": aruco_raw.get("border_width", 0.05),
    }
