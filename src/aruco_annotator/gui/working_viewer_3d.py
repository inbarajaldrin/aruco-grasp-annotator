"""Working 3D viewer widget that integrates Open3D with PyQt6 properly."""

import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QScrollArea
)
from PyQt6.QtGui import QFont

import open3d as o3d


class WorkingViewer3D(QWidget):
    """Working 3D viewer that shows mesh info and provides a launch button for Open3D."""
    
    # Signals
    marker_clicked = pyqtSignal(int)
    grasp_clicked = pyqtSignal(int)
    point_picked = pyqtSignal(tuple)
    
    def __init__(self) -> None:
        super().__init__()
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.markers: Dict[int, Dict[str, Any]] = {}
        self.grasp_poses: Dict[int, Dict[str, Any]] = {}
        self.vis: Optional[o3d.visualization.Visualizer] = None
        
        self.init_ui()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_btn)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.clicked.connect(self.fit_to_view)
        controls_layout.addWidget(self.fit_btn)
        
        self.launch_3d_btn = QPushButton("Launch 3D Viewer")
        self.launch_3d_btn.clicked.connect(self.toggle_3d_viewer)
        self.launch_3d_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        controls_layout.addWidget(self.launch_3d_btn)
        
        controls_layout.addStretch()
        
        # Mode indicator
        self.mode_label = QLabel("Mode: View")
        self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        controls_layout.addWidget(self.mode_label)
        
        layout.addLayout(controls_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left side - 3D view info
        self.view_area = self.create_view_area()
        content_layout.addWidget(self.view_area, 2)
        
        # Right side - Information panel
        self.info_panel = self.create_info_panel()
        content_layout.addWidget(self.info_panel, 1)
        
        layout.addLayout(content_layout)
        
    def create_view_area(self) -> QWidget:
        """Create the 3D view area."""
        view_widget = QFrame()
        view_widget.setFrameStyle(QFrame.Shape.Box)
        view_widget.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 8px;
            }
        """)
        view_widget.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(view_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("3D Model Viewer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Status label
        self.status_label = QLabel("No model loaded")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 14px;
                margin: 20px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Model info
        self.model_info = QTextEdit()
        self.model_info.setMaximumHeight(200)
        self.model_info.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.model_info.setReadOnly(True)
        layout.addWidget(self.model_info)
        
        # Instructions
        instructions = QLabel("""
        <b>Instructions:</b><br>
        1. Load a CAD file using the File menu<br>
        2. Click "Launch 3D Viewer" to open the 3D visualization<br>
        3. Place ArUco markers in the left panel<br>
        4. Define grasp poses in the right panel<br>
        5. Use the 3D viewer to see your annotations in real-time
        """)
        instructions.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 12px;
                margin: 10px;
            }
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        return view_widget
        
    def create_info_panel(self) -> QWidget:
        """Create the information panel."""
        info_widget = QFrame()
        info_widget.setFrameStyle(QFrame.Shape.Box)
        info_widget.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #444;
                border-radius: 8px;
            }
        """)
        info_widget.setMaximumWidth(300)
        
        layout = QVBoxLayout(info_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Model Information")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Statistics
        self.stats_text = QTextEdit()
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        # Markers info
        markers_title = QLabel("ArUco Markers")
        markers_title.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-size: 14px;
                font-weight: bold;
                margin-top: 10px;
            }
        """)
        layout.addWidget(markers_title)
        
        self.markers_text = QTextEdit()
        self.markers_text.setMaximumHeight(100)
        self.markers_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ff6b6b;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.markers_text.setReadOnly(True)
        layout.addWidget(self.markers_text)
        
        # Grasp poses info
        grasps_title = QLabel("Grasp Poses")
        grasps_title.setStyleSheet("""
            QLabel {
                color: #4ecdc4;
                font-size: 14px;
                font-weight: bold;
                margin-top: 10px;
            }
        """)
        layout.addWidget(grasps_title)
        
        self.grasps_text = QTextEdit()
        self.grasps_text.setMaximumHeight(100)
        self.grasps_text.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #4ecdc4;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.grasps_text.setReadOnly(True)
        layout.addWidget(self.grasps_text)
        
        layout.addStretch()
        return info_widget
        
    def load_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """Load a 3D mesh into the viewer."""
        self.mesh = mesh
        
        # Update status
        self.status_label.setText("Model loaded successfully")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 14px;
                margin: 20px;
            }
        """)
        
        # Update model info
        info = self.get_mesh_info(mesh)
        info_text = f"""Model Statistics:
Vertices: {info['vertices']:,}
Triangles: {info['triangles']:,}
Bounding Box:
  Min: ({info['bbox_min'][0]:.3f}, {info['bbox_min'][1]:.3f}, {info['bbox_min'][2]:.3f})
  Max: ({info['bbox_max'][0]:.3f}, {info['bbox_max'][1]:.3f}, {info['bbox_max'][2]:.3f})
Extent: ({info['extent'][0]:.3f}, {info['extent'][1]:.3f}, {info['extent'][2]:.3f})
Volume: {info['volume']:.6f}
Surface Area: {info['surface_area']:.6f}
Has Normals: {info['has_normals']}
Has Colors: {info['has_colors']}
Watertight: {info['is_watertight']}"""
        
        self.model_info.setText(info_text)
        self.update_stats()
        
    def get_mesh_info(self, mesh: o3d.geometry.TriangleMesh) -> dict:
        """Get information about the mesh."""
        bbox = mesh.get_axis_aligned_bounding_box()
        
        # Try to get volume, but handle non-watertight meshes gracefully
        try:
            volume = mesh.get_volume()
        except:
            volume = 0.0  # Volume cannot be computed for non-watertight meshes
        
        # Try to get surface area
        try:
            surface_area = mesh.get_surface_area()
        except:
            surface_area = 0.0
        
        return {
            'vertices': len(mesh.vertices),
            'triangles': len(mesh.triangles),
            'bbox_min': bbox.min_bound.tolist(),
            'bbox_max': bbox.max_bound.tolist(),
            'extent': bbox.get_extent().tolist(),
            'volume': volume,
            'surface_area': surface_area,
            'has_normals': mesh.has_vertex_normals(),
            'has_colors': mesh.has_vertex_colors(),
            'is_watertight': mesh.is_watertight(),
            'is_orientable': mesh.is_orientable()
        }
        
    def toggle_3d_viewer(self) -> None:
        """Toggle the 3D viewer (launch if closed, close if open)."""
        if self.vis is None:
            self.launch_3d_viewer()
        else:
            self.close_3d_viewer()
            
    def close_3d_viewer(self) -> None:
        """Close the 3D viewer."""
        self.cleanup_viewer()
            
    def cleanup_viewer(self) -> None:
        """Clean up the 3D viewer when it's closed."""
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except:
                pass  # Window might already be closed
            self.vis = None
            
            # Update UI
            self.launch_3d_btn.setText("Launch 3D Viewer")
            self.launch_3d_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            
            self.status_label.setText("3D viewer closed")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
            self.update_stats()
            
    def launch_3d_viewer(self) -> None:
        """Launch the Open3D 3D viewer."""
        if self.mesh is None:
            self.status_label.setText("Please load a model first")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
            return
            
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="ArUco Grasp Annotator - 3D View", 
                                  width=800, height=600, visible=True)
            
            # Setup render options
            render_opt = self.vis.get_render_option()
            render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
            render_opt.point_size = 3.0
            render_opt.line_width = 2.0
            render_opt.show_coordinate_frame = True
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(coord_frame)
            
            # Add the mesh
            self.vis.add_geometry(self.mesh)
            
            # Add existing markers
            for marker_id, marker_data in self.markers.items():
                self.add_marker_to_viewer(marker_id, marker_data['position'], marker_data['size'])
                
            # Add existing grasp poses
            for grasp_id, grasp_data in self.grasp_poses.items():
                self.add_grasp_to_viewer(grasp_id, grasp_data['marker_id'], 
                                       grasp_data['position'], grasp_data['orientation'])
            
            # Reset view
            self.vis.reset_view_point(True)
            
            # Update button text and style
            self.launch_3d_btn.setText("Close 3D Viewer")
            self.launch_3d_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
            
            self.status_label.setText("3D viewer launched successfully!")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
            
            # Start the visualization loop
            self.vis.run()
            
            # Clean up when the window is closed (either by user or programmatically)
            self.cleanup_viewer()
            
        except Exception as e:
            self.status_label.setText(f"Error launching 3D viewer: {str(e)}")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
            
    def add_marker_to_viewer(self, marker_id: int, position: tuple, size: float):
        """Add a marker to the 3D viewer."""
        if self.vis is None:
            return
            
        x, y, z = position
        
        # Create marker geometry (cube with ArUco pattern)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/10)
        marker_mesh.translate([x - size/2, y - size/2, z])
        marker_mesh.paint_uniform_color([1, 0, 0])  # Red color
        marker_mesh.compute_vertex_normals()
        
        self.vis.add_geometry(marker_mesh)
        
    def add_grasp_to_viewer(self, grasp_id: int, marker_id: int, position: tuple, orientation: tuple):
        """Add a grasp pose to the 3D viewer."""
        if self.vis is None:
            return
            
        x, y, z = position
        
        # Create coordinate frame for grasp pose
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
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
        arrow.compute_vertex_normals()
        
        self.vis.add_geometry(grasp_frame)
        self.vis.add_geometry(arrow)
        
    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add an ArUco marker visualization."""
        self.markers[marker_id] = {
            'position': position,
            'size': size
        }
        self.update_stats()
        
        # Add to 3D viewer if it's open
        if self.vis is not None:
            self.add_marker_to_viewer(marker_id, position, size)
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from the viewer."""
        if marker_id in self.markers:
            del self.markers[marker_id]
            self.update_stats()
            
    def select_marker(self, marker_id: int) -> None:
        """Highlight the selected marker."""
        self.update_stats()
        
    def add_grasp_pose(self, grasp_id: int, marker_id: int, position: tuple, orientation: tuple) -> None:
        """Add a grasp pose visualization."""
        self.grasp_poses[grasp_id] = {
            'marker_id': marker_id,
            'position': position,
            'orientation': orientation
        }
        self.update_stats()
        
        # Add to 3D viewer if it's open
        if self.vis is not None:
            self.add_grasp_to_viewer(grasp_id, marker_id, position, orientation)
        
    def remove_grasp_pose(self, grasp_id: int) -> None:
        """Remove a grasp pose from the viewer."""
        if grasp_id in self.grasp_poses:
            del self.grasp_poses[grasp_id]
            self.update_stats()
            
    def select_grasp_pose(self, grasp_id: int) -> None:
        """Highlight the selected grasp pose."""
        self.update_stats()
        
    def update_stats(self) -> None:
        """Update the statistics display."""
        # Update markers info
        if self.markers:
            markers_text = "Placed Markers:\n"
            for marker_id, marker in self.markers.items():
                x, y, z = marker['position']
                markers_text += f"  Marker {marker_id}: ({x:.3f}, {y:.3f}, {z:.3f})\n"
        else:
            markers_text = "No markers placed"
        self.markers_text.setText(markers_text)
        
        # Update grasp poses info
        if self.grasp_poses:
            grasps_text = "Defined Grasp Poses:\n"
            for grasp_id, grasp in self.grasp_poses.items():
                x, y, z = grasp['position']
                grasps_text += f"  Grasp {grasp_id} (M{grasp['marker_id']}): ({x:.3f}, {y:.3f}, {z:.3f})\n"
        else:
            grasps_text = "No grasp poses defined"
        self.grasps_text.setText(grasps_text)
        
        # Update main stats
        stats_text = f"""Current Session:
Markers: {len(self.markers)}
Grasp Poses: {len(self.grasp_poses)}

Model Status:
Loaded: {'Yes' if self.mesh is not None else 'No'}
Vertices: {len(self.mesh.vertices) if self.mesh else 0}
Triangles: {len(self.mesh.triangles) if self.mesh else 0}

3D Viewer:
Status: {'Open' if self.vis is not None else 'Closed'}"""
        
        self.stats_text.setText(stats_text)
        
    def set_background(self, bg_type: str) -> None:
        """Set the background color/style."""
        if self.vis is not None:
            render_opt = self.vis.get_render_option()
            if bg_type == "Dark":
                render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
            elif bg_type == "Light":
                render_opt.background_color = np.asarray([0.9, 0.9, 0.9])
            elif bg_type == "Gradient":
                render_opt.background_color = np.asarray([0.2, 0.2, 0.3])
        
    def show_axes(self, show: bool) -> None:
        """Show or hide coordinate axes."""
        pass
        
    def show_grid(self, show: bool) -> None:
        """Show or hide grid."""
        pass
        
    def set_wireframe(self, wireframe: bool) -> None:
        """Set wireframe mode."""
        pass
        
    def reset_camera(self) -> None:
        """Reset camera to default position."""
        if self.vis is not None:
            self.vis.reset_view_point(True)
        self.status_label.setText("Camera reset")
        
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.mesh is not None:
            self.reset_camera()
            self.status_label.setText("Model fitted to view")
        else:
            self.status_label.setText("No model to fit")
            
    def load_annotations(self, data: Dict[str, Any]) -> None:
        """Load annotations from data dictionary."""
        # Clear existing annotations
        self.markers.clear()
        self.grasp_poses.clear()
        
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
            
    def set_interaction_mode(self, mode: str) -> None:
        """Set the interaction mode."""
        self.mode_label.setText(f"Mode: {mode}")
        
        if mode == "view":
            self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        elif mode == "place_marker":
            self.mode_label.setStyleSheet("font-weight: bold; color: red;")
        elif mode == "place_grasp":
            self.mode_label.setStyleSheet("font-weight: bold; color: green;")
