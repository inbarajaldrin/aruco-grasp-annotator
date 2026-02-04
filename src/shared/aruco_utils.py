"""
ArUco marker utilities shared across applications.

Provides functions for generating ArUco markers and related operations.
"""

import cv2
import numpy as np

# ArUco dictionary mapping
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}


def generate_aruco_marker(
    dictionary: str,
    marker_id: int,
    size_pixels: int = 512,
) -> np.ndarray:
    """
    Generate an ArUco marker image.

    Args:
        dictionary: ArUco dictionary name (e.g., "DICT_4X4_50")
        marker_id: Marker ID within the dictionary
        size_pixels: Size of the output marker image in pixels

    Returns:
        Grayscale marker image as numpy array

    Raises:
        ValueError: If dictionary name is unknown
    """
    if dictionary not in ARUCO_DICTS:
        raise ValueError(f"Unknown dictionary: {dictionary}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    return marker_image


def get_aruco_dictionary_max_id(dictionary: str) -> int:
    """
    Get the maximum marker ID for a given ArUco dictionary.

    Args:
        dictionary: ArUco dictionary name

    Returns:
        Maximum valid marker ID

    Raises:
        ValueError: If dictionary name is unknown
    """
    max_ids = {
        "DICT_4X4_50": 49,
        "DICT_4X4_100": 99,
        "DICT_5X5_50": 49,
        "DICT_5X5_100": 99,
        "DICT_6X6_50": 49,
        "DICT_6X6_100": 99,
    }

    if dictionary not in max_ids:
        raise ValueError(f"Unknown dictionary: {dictionary}")

    return max_ids[dictionary]
