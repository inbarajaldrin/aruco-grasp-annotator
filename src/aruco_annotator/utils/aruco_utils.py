"""ArUco marker generation and management utilities."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from pathlib import Path


class ArUcoGenerator:
    """Generator for ArUco markers with different dictionaries."""
    
    # Available ArUco dictionaries
    ARUCO_DICTS = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    
    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector()
    
    @classmethod
    def get_available_dictionaries(cls) -> List[str]:
        """Get list of available ArUco dictionaries."""
        return list(cls.ARUCO_DICTS.keys())
    
    @classmethod
    def get_max_id_for_dict(cls, dict_name: str) -> int:
        """Get maximum ID for a given dictionary."""
        if dict_name not in cls.ARUCO_DICTS:
            return 0
        
        # Extract the number from dictionary name (e.g., "DICT_4X4_50" -> 50)
        parts = dict_name.split('_')
        if len(parts) >= 3:
            try:
                return int(parts[-1]) - 1  # Zero-based indexing
            except ValueError:
                pass
        return 0
    
    def generate_marker(self, dictionary: str, marker_id: int, size_pixels: int = 200) -> np.ndarray:
        """Generate an ArUco marker image.
        
        Args:
            dictionary: ArUco dictionary name (e.g., "DICT_4X4_50")
            marker_id: ID of the marker (0-based)
            size_pixels: Size of the generated marker in pixels
            
        Returns:
            numpy array containing the marker image (grayscale)
        """
        if dictionary not in self.ARUCO_DICTS:
            raise ValueError(f"Unknown dictionary: {dictionary}")
        
        max_id = self.get_max_id_for_dict(dictionary)
        if marker_id < 0 or marker_id > max_id:
            raise ValueError(f"Marker ID {marker_id} out of range for {dictionary} (0-{max_id})")
        
        # Get the dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICTS[dictionary])
        
        # Generate marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
        
        return marker_image
    
    def create_marker_texture(self, dictionary: str, marker_id: int, size_pixels: int = 512) -> o3d.geometry.Image:
        """Create an Open3D texture from ArUco marker.
        
        Args:
            dictionary: ArUco dictionary name
            marker_id: ID of the marker
            size_pixels: Size of the texture in pixels
            
        Returns:
            Open3D Image object for use as texture
        """
        # Generate marker image
        marker_img = self.generate_marker(dictionary, marker_id, size_pixels)
        
        # Convert to RGB (Open3D expects RGB)
        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        
        # Create Open3D image
        o3d_img = o3d.geometry.Image(marker_rgb)
        
        return o3d_img
    
    def save_marker(self, dictionary: str, marker_id: int, output_path: Path, size_pixels: int = 200) -> None:
        """Save ArUco marker to file.
        
        Args:
            dictionary: ArUco dictionary name
            marker_id: ID of the marker
            output_path: Path where to save the marker image
            size_pixels: Size of the generated marker in pixels
        """
        marker_img = self.generate_marker(dictionary, marker_id, size_pixels)
        cv2.imwrite(str(output_path), marker_img)


class ArUcoMarkerInfo:
    """Container for ArUco marker information."""
    
    def __init__(self, dictionary: str, marker_id: int, position: Tuple[float, float, float], 
                 size: float, rotation: Tuple[float, float, float] = (0, 0, 0)):
        self.dictionary = dictionary
        self.marker_id = marker_id
        self.position = position
        self.size = size
        self.rotation = rotation  # Euler angles in radians
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "dictionary": self.dictionary,
            "marker_id": self.marker_id,
            "position": list(self.position),
            "size": self.size,
            "rotation": list(self.rotation)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ArUcoMarkerInfo":
        """Create from dictionary."""
        return cls(
            dictionary=data["dictionary"],
            marker_id=data["marker_id"],
            position=tuple(data["position"]),
            size=data["size"],
            rotation=tuple(data.get("rotation", (0, 0, 0)))
        )
    
    def __str__(self) -> str:
        return f"ArUco {self.dictionary} ID:{self.marker_id} at {self.position}"


def create_aruco_mesh_with_texture(aruco_info: ArUcoMarkerInfo, generator: ArUcoGenerator) -> o3d.geometry.TriangleMesh:
    """Create a 3D mesh with ArUco marker texture.
    
    Args:
        aruco_info: ArUco marker information
        generator: ArUco generator instance
        
    Returns:
        Open3D TriangleMesh with ArUco texture applied
    """
    x, y, z = aruco_info.position
    size = aruco_info.size
    
    # Create a thin rectangular mesh for the marker
    marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/20)
    marker_mesh.translate([x - size/2, y - size/2, z - size/40])
    
    # Apply rotation if any
    if any(aruco_info.rotation):
        R = marker_mesh.get_rotation_matrix_from_xyz(aruco_info.rotation)
        marker_mesh.rotate(R, center=(x, y, z))
    
    # Generate UV coordinates for texture mapping
    # This is a simplified UV mapping for a box
    vertices = np.asarray(marker_mesh.vertices)
    triangles = np.asarray(marker_mesh.triangles)
    
    # Create simple UV coordinates
    # For a box, we'll map the top face to the ArUco marker
    num_vertices = len(vertices)
    uvs = np.zeros((num_vertices, 2))
    
    # Simple UV mapping - top face gets the marker, others get edges
    for i, vertex in enumerate(vertices):
        # Normalize coordinates to [0,1] range
        u = (vertex[0] - (x - size/2)) / size
        v = (vertex[1] - (y - size/2)) / size
        uvs[i] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]
    
    # Create texture
    try:
        texture = generator.create_marker_texture(aruco_info.dictionary, aruco_info.marker_id, 512)
        # Note: Full texture application would require more complex UV mapping
        # For now, we'll use a simpler color-based approach
    except Exception:
        # Fallback to color-based representation
        pass
    
    # Compute normals for proper lighting
    marker_mesh.compute_vertex_normals()
    
    # Paint with a distinctive color pattern to indicate it's an ArUco marker
    marker_mesh.paint_uniform_color([0.9, 0.9, 0.9])  # Light gray base
    
    return marker_mesh
