"""Annotation manager for handling markers and grasp poses."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class AnnotationManager:
    """Manager for handling annotations data and serialization."""
    
    def __init__(self):
        self.model_file: Optional[Path] = None
        self.annotations: Dict[str, Any] = {}
        self.has_changes = False
        
    def set_model_file(self, file_path: Path) -> None:
        """Set the current model file."""
        self.model_file = file_path
        self.annotations = {
            "model_file": str(file_path),
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "markers": [],
            "grasp_poses": []
        }
        self.has_changes = False
        
    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add a marker to the annotations."""
        marker_data = {
            "id": marker_id,
            "position": list(position),
            "size": size,
            "created_at": datetime.now().isoformat()
        }
        
        # Remove existing marker with same ID
        self.annotations["markers"] = [
            m for m in self.annotations["markers"] if m["id"] != marker_id
        ]
        
        # Add new marker
        self.annotations["markers"].append(marker_data)
        self._mark_modified()
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from annotations."""
        # Remove marker
        self.annotations["markers"] = [
            m for m in self.annotations["markers"] if m["id"] != marker_id
        ]
        
        # Remove associated grasp poses
        self.annotations["grasp_poses"] = [
            g for g in self.annotations["grasp_poses"] if g["marker_id"] != marker_id
        ]
        
        self._mark_modified()
        
    def add_grasp_pose(self, grasp_id: int, name: str, marker_id: int, 
                      position: tuple, orientation: tuple) -> None:
        """Add a grasp pose to the annotations."""
        grasp_data = {
            "id": grasp_id,
            "name": name,
            "marker_id": marker_id,
            "position": list(position),
            "orientation": list(orientation),  # quaternion (w, x, y, z)
            "created_at": datetime.now().isoformat()
        }
        
        # Remove existing grasp with same ID
        self.annotations["grasp_poses"] = [
            g for g in self.annotations["grasp_poses"] if g["id"] != grasp_id
        ]
        
        # Add new grasp
        self.annotations["grasp_poses"].append(grasp_data)
        self._mark_modified()
        
    def remove_grasp_pose(self, grasp_id: int) -> None:
        """Remove a grasp pose from annotations."""
        self.annotations["grasp_poses"] = [
            g for g in self.annotations["grasp_poses"] if g["id"] != grasp_id
        ]
        self._mark_modified()
        
    def update_grasp_pose(self, grasp_id: int, position: tuple, orientation: tuple) -> None:
        """Update a grasp pose position and orientation."""
        for grasp in self.annotations["grasp_poses"]:
            if grasp["id"] == grasp_id:
                grasp["position"] = list(position)
                grasp["orientation"] = list(orientation)
                grasp["modified_at"] = datetime.now().isoformat()
                break
        self._mark_modified()
        
    def export_annotations(self, output_path: Path, markers: List[Dict], grasp_poses: List[Dict]) -> None:
        """Export annotations to JSON file."""
        # Update annotations with current data
        self.annotations["markers"] = markers
        self.annotations["grasp_poses"] = grasp_poses
        self.annotations["modified_at"] = datetime.now().isoformat()
        
        # Add metadata
        export_data = {
            **self.annotations,
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "total_markers": len(markers),
            "total_grasp_poses": len(grasp_poses)
        }
        
        # Convert numpy arrays to lists (if any)
        export_data = self._serialize_numpy(export_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.has_changes = False
        
    def import_annotations(self, input_path: Path) -> Dict[str, Any]:
        """Import annotations from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        # Validate data structure
        self._validate_annotations(data)
        
        # Update internal state
        self.annotations = data
        self.has_changes = False
        
        return data
        
    def get_markers_for_export(self) -> List[Dict[str, Any]]:
        """Get markers formatted for robotics pipeline export."""
        exported_markers = []
        
        for marker in self.annotations.get("markers", []):
            # Get associated grasp poses
            associated_grasps = [
                g for g in self.annotations.get("grasp_poses", []) 
                if g["marker_id"] == marker["id"]
            ]
            
            # Format grasp poses
            formatted_grasps = []
            for grasp in associated_grasps:
                formatted_grasps.append({
                    "name": grasp["name"],
                    "position": grasp["position"],
                    "orientation": grasp["orientation"],
                    "approach_vector": self._compute_approach_vector(grasp["orientation"])
                })
                
            exported_markers.append({
                "id": marker["id"],
                "position": marker["position"],
                "orientation": [1.0, 0.0, 0.0, 0.0],  # Identity quaternion for marker
                "size": marker.get("size", 0.05),
                "grasp_poses": formatted_grasps
            })
            
        return exported_markers
        
    def export_for_robotics(self, output_path: Path) -> None:
        """Export annotations in format suitable for robotics pipeline."""
        robotics_data = {
            "object_file": str(self.model_file) if self.model_file else "",
            "created_at": datetime.now().isoformat(),
            "coordinate_frame": "object_frame",
            "units": "meters",
            "markers": self.get_markers_for_export()
        }
        
        with open(output_path, 'w') as f:
            json.dump(robotics_data, f, indent=2)
            
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self.has_changes
        
    def _mark_modified(self) -> None:
        """Mark annotations as modified."""
        self.has_changes = True
        if "modified_at" in self.annotations:
            self.annotations["modified_at"] = datetime.now().isoformat()
            
    def _serialize_numpy(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy(item) for item in obj]
        else:
            return obj
            
    def _validate_annotations(self, data: Dict[str, Any]) -> None:
        """Validate annotation data structure."""
        required_fields = ["markers", "grasp_poses"]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
        # Validate markers
        for marker in data["markers"]:
            required_marker_fields = ["id", "position"]
            for field in required_marker_fields:
                if field not in marker:
                    raise ValueError(f"Marker missing required field: {field}")
                    
        # Validate grasp poses
        for grasp in data["grasp_poses"]:
            required_grasp_fields = ["id", "name", "marker_id", "position", "orientation"]
            for field in required_grasp_fields:
                if field not in grasp:
                    raise ValueError(f"Grasp pose missing required field: {field}")
                    
    def _compute_approach_vector(self, quaternion: List[float]) -> List[float]:
        """Compute approach vector from quaternion orientation."""
        # This is a simplified computation
        # In practice, you'd use proper quaternion math
        # For now, return a default approach vector (positive Z)
        return [0.0, 0.0, 1.0]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about current annotations."""
        markers = self.annotations.get("markers", [])
        grasps = self.annotations.get("grasp_poses", [])
        
        # Grasp poses per marker
        grasps_per_marker = {}
        for grasp in grasps:
            marker_id = grasp["marker_id"]
            grasps_per_marker[marker_id] = grasps_per_marker.get(marker_id, 0) + 1
            
        return {
            "total_markers": len(markers),
            "total_grasp_poses": len(grasps),
            "grasps_per_marker": grasps_per_marker,
            "has_unsaved_changes": self.has_changes,
            "model_file": str(self.model_file) if self.model_file else None
        }
