"""ArUco data transformation service for Assembly App."""

from typing import Any


def transform_aruco_data(aruco_raw: dict) -> dict | None:
    """Transform ArUco data from new format (T_object_to_marker) to frontend format (pose_absolute).

    Args:
        aruco_raw: Raw ArUco data from JSON file with T_object_to_marker format

    Returns:
        Transformed ArUco data with pose_absolute format for frontend, or None if input is None
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

        if "T_object_to_marker" not in marker:
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} missing 'T_object_to_marker'. "
                f"Marker keys: {list(marker.keys())}"
            )

        if not isinstance(marker["T_object_to_marker"], dict):
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} has invalid "
                f"'T_object_to_marker' type: {type(marker['T_object_to_marker'])}"
            )

        transformed_marker = {
            "aruco_id": marker.get("aruco_id"),
            "size": marker.get("size", default_size),
            "face_type": marker.get("face_type"),
            "surface_normal": marker.get("surface_normal"),
        }

        t_obj_to_marker = marker["T_object_to_marker"]
        if "position" not in t_obj_to_marker or "rotation" not in t_obj_to_marker:
            raise ValueError(
                f"Marker {marker.get('aruco_id', f'at index {idx}')} T_object_to_marker "
                f"missing 'position' or 'rotation'. Keys: {list(t_obj_to_marker.keys())}"
            )

        transformed_marker["pose_absolute"] = {
            "position": t_obj_to_marker.get("position", {"x": 0, "y": 0, "z": 0}),
            "rotation": t_obj_to_marker.get(
                "rotation", {"roll": 0, "pitch": 0, "yaw": 0}
            ),
        }

        transformed_markers.append(transformed_marker)

    return {
        "markers": transformed_markers,
        "aruco_dictionary": aruco_raw.get("aruco_dictionary", "DICT_4X4_50"),
        "size": default_size,
        "border_width": aruco_raw.get("border_width", 0.05),
    }
