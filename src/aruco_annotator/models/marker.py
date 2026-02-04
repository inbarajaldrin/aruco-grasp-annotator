"""MarkerData model for ArUco marker placement and tracking."""

import warnings
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class MarkerData:
    """Enhanced marker data with separated geometric and rotation info."""

    def __init__(
        self,
        aruco_id: int,
        dictionary: str,
        size: float,
        border_width: float,
        position: tuple,
        face_normal: tuple,
        face_type: str,
    ) -> None:
        self.aruco_id = aruco_id
        self.dictionary = dictionary
        self.size = size
        self.border_width = border_width

        # Geometric properties (INVARIANT - don't change with in-plane rotation)
        # Position and normal are stored in OBJECT LOCAL coordinate system
        self.position = np.array(position)
        self.face_normal = np.array(face_normal) / (np.linalg.norm(face_normal) + 1e-8)
        self.face_type = face_type

        # Calculate base orientation (marker Z+ aligned with face normal)
        self.base_rotation_matrix = self._calculate_base_rotation()

        # In-plane rotation offset (rotation around face normal)
        self.in_plane_rotation_deg = 0.0

        # In-plane translation offset (translation along marker's X and Y axes)
        self.in_plane_translation = np.array([0.0, 0.0])

        # Store initial position for translation calculations
        self.initial_position = np.array(position).copy()

    def _calculate_base_rotation(self) -> np.ndarray:
        """Calculate rotation matrix that aligns marker Z+ with face normal."""
        z_world = np.array([0.0, 0.0, 1.0])
        normal = self.face_normal

        if np.allclose(normal, z_world, atol=1e-6):
            return np.eye(3)
        if np.allclose(normal, -z_world, atol=1e-6):
            return Rotation.from_euler("x", 180, degrees=True).as_matrix()

        rotation_axis = np.cross(z_world, normal)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        rotation_angle = np.arccos(np.clip(np.dot(z_world, normal), -1.0, 1.0))

        return Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

    def get_current_rotation_matrix(self) -> np.ndarray:
        """Get current rotation matrix including in-plane rotation."""
        R_base = self.base_rotation_matrix

        if abs(self.in_plane_rotation_deg) < 1e-6:
            return R_base

        R_inplane_marker = Rotation.from_euler(
            "z", self.in_plane_rotation_deg, degrees=True
        ).as_matrix()

        return R_base @ R_inplane_marker

    def set_in_plane_rotation(self, degrees: float) -> None:
        """Set in-plane rotation in degrees."""
        self.in_plane_rotation_deg = degrees % 360.0

    def _get_quaternion_from_normal(self) -> tuple:
        """
        Get quaternion for primary axis faces based on the actual stored normal.
        Returns (quaternion, is_primary_axis) tuple.
        Quaternion format: [x, y, z, w]
        """
        normal = self.face_normal
        in_plane_rad = np.deg2rad(self.in_plane_rotation_deg)

        is_primary = (
            np.any(np.allclose(normal, [1, 0, 0], atol=1e-6))
            or np.any(np.allclose(normal, [-1, 0, 0], atol=1e-6))
            or np.any(np.allclose(normal, [0, 1, 0], atol=1e-6))
            or np.any(np.allclose(normal, [0, -1, 0], atol=1e-6))
            or np.any(np.allclose(normal, [0, 0, 1], atol=1e-6))
            or np.any(np.allclose(normal, [0, 0, -1], atol=1e-6))
        )

        if not is_primary:
            return None, False

        z_world = np.array([0.0, 0.0, 1.0])

        if np.allclose(normal, z_world, atol=1e-6):
            R_base = np.eye(3)
        elif np.allclose(normal, -z_world, atol=1e-6):
            R_base = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        else:
            rotation_axis = np.cross(z_world, normal)
            rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
            rotation_angle = np.arccos(np.clip(np.dot(z_world, normal), -1.0, 1.0))
            R_base = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

        rot_base = Rotation.from_matrix(R_base)
        quat_base = rot_base.as_quat()

        if abs(in_plane_rad) > 1e-6:
            rot_inplane = Rotation.from_euler("z", in_plane_rad, degrees=False)
            rot_combined = Rotation.from_quat(quat_base) * rot_inplane
            quat = rot_combined.as_quat()
        else:
            quat = quat_base

        return quat, True

    def _rotation_matrix_to_euler_avoiding_gimbal_lock(
        self, R: np.ndarray
    ) -> np.ndarray:
        """Convert rotation matrix to Euler angles using quaternion as intermediate."""
        rot_scipy = Rotation.from_matrix(R)
        quat = rot_scipy.as_quat()
        rot_from_quat = Rotation.from_quat(quat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            euler = rot_from_quat.as_euler("xyz")

        return euler

    def get_T_object_to_marker(self, cad_center_local: list) -> dict[str, Any]:
        """
        Calculate T_object_to_marker (includes full rotation: base + in-plane).

        Args:
            cad_center_local: Object center in object local space

        Returns:
            T_object_to_marker: Transform from object frame to marker frame
        """
        cad_center_local = np.array(cad_center_local)
        vec_object_to_marker = self.position - cad_center_local

        quat, is_primary = self._get_quaternion_from_normal()

        if is_primary:
            normal = self.face_normal
            in_plane_rad = np.deg2rad(self.in_plane_rotation_deg)

            if np.allclose(normal, [1, 0, 0], atol=1e-6):
                euler = np.array([0.0, np.pi / 2, in_plane_rad])
            elif np.allclose(normal, [-1, 0, 0], atol=1e-6):
                euler = np.array([0.0, -np.pi / 2, in_plane_rad])
            elif np.allclose(normal, [0, 1, 0], atol=1e-6):
                euler = np.array([-np.pi / 2, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, -1, 0], atol=1e-6):
                euler = np.array([np.pi / 2, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, 0, 1], atol=1e-6):
                euler = np.array([0.0, 0.0, in_plane_rad])
            elif np.allclose(normal, [0, 0, -1], atol=1e-6):
                euler = np.array([np.pi, 0.0, in_plane_rad])
            else:
                rot_from_quat = Rotation.from_quat(quat)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    euler = rot_from_quat.as_euler("xyz")
                euler[2] = in_plane_rad
        else:
            R_object_to_marker_full = self.get_current_rotation_matrix().T
            rot_scipy = Rotation.from_matrix(R_object_to_marker_full)
            quat = rot_scipy.as_quat()
            euler = self._rotation_matrix_to_euler_avoiding_gimbal_lock(
                R_object_to_marker_full
            )

        return {
            "position": {
                "x": float(vec_object_to_marker[0]),
                "y": float(vec_object_to_marker[1]),
                "z": float(vec_object_to_marker[2]),
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
        }
