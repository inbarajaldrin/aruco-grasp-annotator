"""3D viewer widget using Open3D for CAD model visualization."""

import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class Viewer3D(QWidget):
    """3D viewer widget for CAD models and annotations."""
    
    # Signals
    marker_clicked = pyqtSignal(int)  # marker_id
    grasp_clicked = pyqtSignal(int)   # grasp_id
    point_picked = pyqtSignal(tuple)  # (x, y, z)
    
    def __init__(self) -> None:
        super().__init__()
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.grasp_poses: Dict[int, Dict[str, Any]] = {}
        self.coordinate_frame: Optional[o3d.geometry.TriangleMesh] = None
        
        self.init_ui()
        self.setup_viewer()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Open3D visualizer widget
        self.setup_open3d_widget()
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_btn)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.clicked.connect(self.fit_to_view)
        controls_layout.addWidget(self.fit_btn)
        
        controls_layout.addStretch()
        
        # Mode indicator
        self.mode_label = QLabel("Mode: View")
        self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        controls_layout.addWidget(self.mode_label)
        
        layout.addLayout(controls_layout)
        
    def setup_open3d_widget(self) -> None:
        """Setup the Open3D visualization widget."""
        # For now, we'll use a placeholder widget
        # In a full implementation, you'd integrate Open3D's GUI system
        self.viewer_widget = QWidget()
        self.viewer_widget.setMinimumSize(600, 400)
        self.viewer_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        
        # Add a label indicating this is the 3D view area
        view_layout = QVBoxLayout(self.viewer_widget)
        view_label = QLabel("3D Viewer\\n\\nLoad a CAD file to begin")
        view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 16px;
                border: none;
            }
        """)
        view_layout.addWidget(view_label)
        
        self.layout().addWidget(self.viewer_widget)
        
        # Initialize Open3D visualizer (headless for now)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)
        
        # Setup default view
        self.setup_default_view()
        
    def setup_viewer(self) -> None:
        """Setup the 3D viewer with default settings."""
        # Add coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coordinate_frame)
        
        # Setup lighting and camera
        self.reset_camera()
        
    def setup_default_view(self) -> None:
        """Setup default view settings."""
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
        render_opt.point_size = 3.0
        render_opt.line_width = 2.0
        render_opt.show_coordinate_frame = True
        
    def load_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """Load a 3D mesh into the viewer."""
        # Remove previous mesh if exists
        if self.mesh is not None:
            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
            
        # Prepare mesh (Open3D doesn't have copy method, create a new mesh)
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = mesh.vertices
        self.mesh.triangles = mesh.triangles
        if mesh.has_vertex_normals():
            self.mesh.vertex_normals = mesh.vertex_normals
        if mesh.has_vertex_colors():
            self.mesh.vertex_colors = mesh.vertex_colors
        
        # Compute normals if not present
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
            
        # Set default color
        if not self.mesh.has_vertex_colors():
            self.mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
        # Add to visualizer
        self.vis.add_geometry(self.mesh)
        
        # Update the placeholder widget
        if hasattr(self.viewer_widget, 'layout'):
            # Clear the placeholder label
            for i in reversed(range(self.viewer_widget.layout().count())): 
                self.viewer_widget.layout().itemAt(i).widget().setParent(None)
                
            # Add mesh info
            info_label = QLabel(f"Loaded mesh:\\n{len(self.mesh.vertices)} vertices\\n{len(self.mesh.triangles)} faces")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    border: none;
                }
            """)
            self.viewer_widget.layout().addWidget(info_label)
        
        # Fit to view
        self.fit_to_view()
        
    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add an ArUco marker visualization at the specified position."""
        x, y, z = position
        
        # Create marker geometry (cube with ArUco pattern)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/10)
        marker_mesh.translate([x - size/2, y - size/2, z])
        marker_mesh.paint_uniform_color([1, 0, 0])  # Red color
        
        # Store marker
        self.markers[marker_id] = {
            'geometry': marker_mesh,
            'position': position,
            'size': size
        }
        
        # Add to visualizer
        self.vis.add_geometry(marker_mesh)
        self.vis.update_geometry(marker_mesh)
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from the viewer."""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            del self.markers[marker_id]
            
    def select_marker(self, marker_id: int) -> None:
        """Highlight the selected marker."""
        # Reset all markers to red
        for mid, marker in self.markers.items():
            color = [1, 1, 0] if mid == marker_id else [1, 0, 0]  # Yellow if selected, red otherwise
            marker['geometry'].paint_uniform_color(color)
            self.vis.update_geometry(marker['geometry'])
            
    def add_grasp_pose(self, grasp_id: int, marker_id: int, position: tuple, orientation: tuple) -> None:
        """Add a grasp pose visualization."""
        x, y, z = position
        
        # Create coordinate frame for grasp pose
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        
        # Apply transformation (simplified - in real implementation, use quaternion)
        grasp_frame.translate([x, y, z])
        
        # Create approach vector (arrow)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002,
            cone_radius=0.005,
            cylinder_height=0.03,
            cone_height=0.01
        )
        arrow.translate([x, y, z + 0.02])
        arrow.paint_uniform_color([0, 1, 0])  # Green
        
        # Store grasp pose
        self.grasp_poses[grasp_id] = {
            'frame': grasp_frame,
            'arrow': arrow,
            'marker_id': marker_id,
            'position': position,
            'orientation': orientation
        }
        
        # Add to visualizer
        self.vis.add_geometry(grasp_frame)
        self.vis.add_geometry(arrow)
        
    def remove_grasp_pose(self, grasp_id: int) -> None:
        """Remove a grasp pose from the viewer."""
        if grasp_id in self.grasp_poses:
            grasp = self.grasp_poses[grasp_id]
            self.vis.remove_geometry(grasp['frame'], reset_bounding_box=False)
            self.vis.remove_geometry(grasp['arrow'], reset_bounding_box=False)
            del self.grasp_poses[grasp_id]
            
    def select_grasp_pose(self, grasp_id: int) -> None:
        """Highlight the selected grasp pose."""
        for gid, grasp in self.grasp_poses.items():
            # Change arrow color based on selection
            color = [1, 1, 0] if gid == grasp_id else [0, 1, 0]  # Yellow if selected
            grasp['arrow'].paint_uniform_color(color)
            self.vis.update_geometry(grasp['arrow'])
            
    def set_background(self, bg_type: str) -> None:
        """Set the background color/style."""
        render_opt = self.vis.get_render_option()
        
        if bg_type == "Dark":
            render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
        elif bg_type == "Light":
            render_opt.background_color = np.asarray([0.9, 0.9, 0.9])
        elif bg_type == "Gradient":
            render_opt.background_color = np.asarray([0.2, 0.2, 0.3])
            
    def show_axes(self, show: bool) -> None:
        """Show or hide coordinate axes."""
        if show and self.coordinate_frame not in self.vis.get_geometry():
            self.vis.add_geometry(self.coordinate_frame)
        elif not show and self.coordinate_frame in self.vis.get_geometry():
            self.vis.remove_geometry(self.coordinate_frame, reset_bounding_box=False)
            
    def show_grid(self, show: bool) -> None:
        """Show or hide grid (placeholder for now)."""
        # In a full implementation, you'd add/remove a grid geometry
        pass
        
    def set_wireframe(self, wireframe: bool) -> None:
        """Set wireframe mode."""
        if self.mesh is not None:
            render_opt = self.vis.get_render_option()
            if wireframe:
                render_opt.mesh_show_wireframe = True
                render_opt.mesh_show_back_face = True
            else:
                render_opt.mesh_show_wireframe = False
                render_opt.mesh_show_back_face = False
                
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self.vis.reset_view_point(True)
        
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.mesh is not None:
            # Get the bounding box and set camera to look at it
            bbox = self.mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent()
            max_extent = np.max(extent)
            
            # Set camera parameters to fit the model
            view_control = self.vis.get_view_control()
            view_control.set_lookat(center)
            view_control.set_zoom(0.8)  # Adjust zoom to fit
            
    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotations from data dictionary."""
        # Clear existing annotations
        for marker_id in list(self.markers.keys()):
            self.remove_marker(marker_id)
        for grasp_id in list(self.grasp_poses.keys()):
            self.remove_grasp_pose(grasp_id)
            
        # Load markers
        for marker_data in data.get("markers", []):
            self.add_marker(
                marker_data["id"],
                tuple(marker_data["position"]),
                marker_data.get("size", 0.05)
            )
            
        # Load grasp poses
        for grasp_data in data.get("grasp_poses", []):
            self.add_grasp_pose(
                grasp_data["id"],
                grasp_data["marker_id"],
                tuple(grasp_data["position"]),
                tuple(grasp_data["orientation"])
            )
            
    def get_pick_point(self, x: int, y: int) -> Optional[tuple]:
        """Get the 3D point at screen coordinates (for future implementation)."""
        # This would involve ray casting into the scene
        # For now, return None
        return None
        
    def set_interaction_mode(self, mode: str) -> None:
        """Set the interaction mode (view, place_marker, place_grasp)."""
        self.mode_label.setText(f"Mode: {mode}")
        
        if mode == "view":
            self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        elif mode == "place_marker":
            self.mode_label.setStyleSheet("font-weight: bold; color: red;")
        elif mode == "place_grasp":
            self.mode_label.setStyleSheet("font-weight: bold; color: green;")
