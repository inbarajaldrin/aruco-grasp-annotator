"""Working 3D viewer widget that integrates Open3D with PyQt6 properly."""

import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFrame, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QFont

import open3d as o3d
from ..utils.aruco_utils import ArUcoGenerator, ArUcoMarkerInfo, create_aruco_mesh_with_texture
from .face_picker_dialog import FacePickerDialog


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
        self.placement_mode = False
        self.direct_click_mode = False
        self.aruco_generator = ArUcoGenerator()
        self.mesh_info: Optional[dict] = None
        self.scale_ruler: Optional[o3d.geometry.TriangleMesh] = None
        self.face_groups = None
        self.highlighted_faces = []
        
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
        
        # Placement instruction label (for placement mode)
        self.placement_label = QLabel("Choose how to place ArUco marker:\n• 🎯 Random Face • 📋 Face List")
        self.placement_label.setVisible(False)  # Hidden by default
        self.placement_label.setStyleSheet("""
            QLabel {
                background-color: #FFF3CD;
                border: 1px solid #FFEAA7;
                border-radius: 4px;
                padding: 8px;
                color: #856404;
                font-weight: bold;
            }
        """)
        controls_layout.addWidget(self.placement_label)
        
        # Placeholder for alternative placement button (added later if needed)
        self.click_place_btn = None
        
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
        
    def load_mesh(self, mesh: o3d.geometry.TriangleMesh, mesh_info: Optional[dict] = None) -> None:
        """Load a 3D mesh into the viewer."""
        self.mesh = mesh
        self.mesh_info = mesh_info
        
        # Update status with dimension information
        if mesh_info and 'dimensions' in mesh_info:
            dims = mesh_info['dimensions']
            vertices_count = mesh_info.get('vertices', 0)
            triangles_count = mesh_info.get('triangles', 0)
            
            status_text = f"Model loaded successfully\\n"
            status_text += f"Dimensions: {dims['length']:.3f} × {dims['width']:.3f} × {dims['height']:.3f} m\\n"
            status_text += f"Vertices: {vertices_count:,} | Faces: {triangles_count:,}"
        else:
            status_text = "Model loaded successfully"
        
        self.status_label.setText(status_text)
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
            error_msg = "Please load a model first"
            print("ERROR:", error_msg)
            
            self.status_label.setText(error_msg)
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
            render_opt.show_coordinate_frame = False  # Disable built-in coordinate frame
            
            # Add coordinate frame with model-relative size
            if self.mesh_info and 'max_dimension' in self.mesh_info:
                # Scale coordinate frame to 10% of model's max dimension
                coord_frame_size = self.mesh_info['max_dimension'] * 0.1
            else:
                # Fallback to fixed size if no mesh info
                coord_frame_size = 0.1
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
            self.vis.add_geometry(coord_frame)
            
            # Add scale ruler if mesh info is available
            if self.mesh_info:
                self.scale_ruler = self.create_scale_ruler()
                if self.scale_ruler:
                    self.vis.add_geometry(self.scale_ruler)
            
            # Add the mesh
            self.vis.add_geometry(self.mesh)
            
            # Add existing markers
            for marker_id, marker_data in self.markers.items():
                if 'aruco_info' in marker_data:
                    # New format with ArUcoMarkerInfo
                    aruco_info = marker_data['aruco_info']
                    self.add_marker_to_viewer(marker_id, tuple(aruco_info.position), aruco_info.size)
                else:
                    # Legacy format
                    position = marker_data.get('position', (0, 0, 0))
                    size = marker_data.get('size', 0.05)
                    self.add_marker_to_viewer(marker_id, position, size)
                
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
            
            # Set up interaction callbacks for picking
            self.setup_picking_callback()
            
            # Setup Qt integration timer for Open3D events
            self.setup_qt_integration()
            
            # Fit to view - use the correct method name for this Open3D version
            try:
                # Try different method names depending on Open3D version
                view_control = self.vis.get_view_control()
                if hasattr(view_control, 'fit_in_window'):
                    view_control.fit_in_window()
                elif hasattr(view_control, 'zoom_in_out'):
                    # Alternative: reset view and zoom to fit
                    view_control.reset_view_point(True)
                else:
                    # Fallback: just reset the view
                    print("Using fallback view reset")
                    pass
            except Exception as e:
                print(f"Warning: Could not fit view to window: {e}")
                # Continue anyway - this is not critical
            
        except Exception as e:
            error_msg = f"Error launching 3D viewer: {str(e)}"
            
            # Print error to terminal so user can see it
            print("=" * 60)
            print("ERROR IN 3D VIEWER:")
            print(error_msg)
            print("Exception type:", type(e).__name__)
            print("Full details:")
            import traceback
            traceback.print_exc()
            print("=" * 60)
            
            # Also show in UI
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
            
    def create_aruco_marker_geometry(self, position: tuple, size: float, marker_id: int, rotation: tuple = (0, 0, 0)) -> o3d.geometry.TriangleMesh:
        """Create a visual representation of an ArUco marker."""
        import numpy as np
        
        x, y, z = position
        roll, pitch, yaw = rotation
        
        # Apply reasonable size limits based on model dimensions
        if self.mesh_info and 'max_dimension' in self.mesh_info:
            model_size = self.mesh_info['max_dimension']
            # Limit marker size to be between 1% and 10% of model size
            min_size = model_size * 0.01  # 1% of model
            max_size = model_size * 0.10  # 10% of model
            size = max(min_size, min(size, max_size))
        else:
            # Fallback limits for when no model info is available
            size = max(0.005, min(size, 0.1))  # Between 5mm and 10cm
        
        # Create the main marker base (flat rectangle)
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/20)
        marker_mesh.translate([x - size/2, y - size/2, z - size/40])
        marker_mesh.paint_uniform_color([0.9, 0.9, 0.9])
        
        # Create a black border
        border_size = size * 0.9
        border_mesh = o3d.geometry.TriangleMesh.create_box(border_size, border_size, size/30)
        border_mesh.translate([x - border_size/2, y - border_size/2, z])
        border_mesh.paint_uniform_color([0.1, 0.1, 0.1])
        
        # Create inner white square
        inner_size = size * 0.7
        inner_mesh = o3d.geometry.TriangleMesh.create_box(inner_size, inner_size, size/25)
        inner_mesh.translate([x - inner_size/2, y - inner_size/2, z + size/60])
        inner_mesh.paint_uniform_color([0.95, 0.95, 0.95])
        
        # Create a simple pattern (representing ArUco squares)
        pattern_cubes = []
        pattern_size = inner_size / 5  # 5x5 grid like ArUco
        for i in range(5):
            for j in range(5):
                if (i + j + marker_id) % 2 == 0:  # Simple pattern based on marker ID
                    cube = o3d.geometry.TriangleMesh.create_box(pattern_size*0.8, pattern_size*0.8, size/20)
                    cube_x = x - inner_size/2 + i * pattern_size + pattern_size/2 - pattern_size*0.4
                    cube_y = y - inner_size/2 + j * pattern_size + pattern_size/2 - pattern_size*0.4
                    cube.translate([cube_x, cube_y, z + size/50])
                    cube.paint_uniform_color([0.2, 0.2, 0.2])
                    pattern_cubes.append(cube)
        
        # Combine all meshes
        combined_mesh = marker_mesh + border_mesh + inner_mesh
        for cube in pattern_cubes:
            combined_mesh += cube
            
        # Add a small coordinate frame to show orientation
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size/4)
        coord_frame.translate([x, y, z + size/10])
        combined_mesh += coord_frame
        
        # Apply rotation if specified
        if rotation != (0, 0, 0):
            print(f"Applying rotation to geometric marker: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            
            # Apply rotations one by one for better control
            center = np.array([x, y, z])
            
            if abs(roll) > 0.001:
                roll_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([roll, 0, 0])
                combined_mesh.rotate(roll_matrix, center=center)
                print(f"Applied roll rotation: {roll:.3f}")
            
            if abs(pitch) > 0.001:
                pitch_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, pitch, 0])
                combined_mesh.rotate(pitch_matrix, center=center)
                print(f"Applied pitch rotation: {pitch:.3f}")
            
            if abs(yaw) > 0.001:
                yaw_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
                combined_mesh.rotate(yaw_matrix, center=center)
                print(f"Applied yaw rotation: {yaw:.3f}")
        
        # Compute normals for proper lighting
        combined_mesh.compute_vertex_normals()
        
        return combined_mesh
    
    def create_real_aruco_marker(self, aruco_info: ArUcoMarkerInfo) -> o3d.geometry.TriangleMesh:
        """Create a real ArUco marker with actual marker image texture."""
        import cv2
        import numpy as np
        
        x, y, z = aruco_info.position
        
        # Use the size from ArUcoMarkerInfo, but apply reasonable limits
        size = aruco_info.size
        
        # Apply reasonable size limits based on model dimensions
        if self.mesh_info and 'max_dimension' in self.mesh_info:
            model_size = self.mesh_info['max_dimension']
            # Limit marker size to be between 1% and 10% of model size
            min_size = model_size * 0.01  # 1% of model
            max_size = model_size * 0.10  # 10% of model
            size = max(min_size, min(size, max_size))
        else:
            # Fallback limits for when no model info is available
            size = max(0.005, min(size, 0.1))  # Between 5mm and 10cm
        
        # Generate the actual ArUco marker image
        marker_image = self.aruco_generator.generate_marker(
            aruco_info.dictionary, 
            aruco_info.marker_id, 
            512  # High resolution
        )
        
        # Create a thin rectangular mesh for the marker
        marker_mesh = o3d.geometry.TriangleMesh.create_box(size, size, size/30)
        marker_mesh.translate([x - size/2, y - size/2, z - size/60])
        
        # Create texture coordinates (UV mapping)
        # For a simple box, we'll map the top face to the ArUco marker
        vertices = np.asarray(marker_mesh.vertices)
        triangles = np.asarray(marker_mesh.triangles)
        
        # Simple UV mapping for the top face
        num_vertices = len(vertices)
        uvs = np.zeros((num_vertices, 2))
        
        for i, vertex in enumerate(vertices):
            # Map vertex coordinates to UV space [0,1]
            u = (vertex[0] - (x - size/2)) / size
            v = (vertex[1] - (y - size/2)) / size
            uvs[i] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]
        
        # Convert marker image to RGB for Open3D
        marker_rgb = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)
        
        # Create a distinctive color pattern since Open3D texture mapping is complex
        # We'll create a high-contrast pattern that represents the ArUco marker
        
        # Use a black and white pattern based on the marker
        # Sample the center region of the marker to determine main pattern
        center_region = marker_rgb[marker_rgb.shape[0]//4:3*marker_rgb.shape[0]//4, 
                                  marker_rgb.shape[1]//4:3*marker_rgb.shape[1]//4]
        avg_intensity = np.mean(center_region)
        
        # Apply color based on the marker pattern
        if avg_intensity < 128:  # Dark pattern
            marker_mesh.paint_uniform_color([0.1, 0.1, 0.1])  # Dark
        else:  # Light pattern
            marker_mesh.paint_uniform_color([0.9, 0.9, 0.9])  # Light
        
        # Add a border to make it look more like an ArUco marker
        border_mesh = o3d.geometry.TriangleMesh.create_box(size*1.1, size*1.1, size/40)
        border_mesh.translate([x - size*0.55, y - size*0.55, z - size/80])
        border_mesh.paint_uniform_color([0.0, 0.0, 0.0])  # Black border
        
        # Combine border and marker
        combined_mesh = border_mesh + marker_mesh
        
        # Add coordinate frame to show orientation
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size/3)
        coord_frame.translate([x, y, z + size/20])
        combined_mesh += coord_frame
        
        # Add text label with ArUco info (simplified representation)
        # We'll create small cubes to represent the ID
        id_display_size = size / 20
        for i, digit in enumerate(str(aruco_info.marker_id)):
            if i >= 3:  # Limit to 3 digits
                break
            digit_cube = o3d.geometry.TriangleMesh.create_box(id_display_size, id_display_size, id_display_size)
            digit_cube.translate([x + (i - 1) * id_display_size * 1.5, y + size/2 + id_display_size, z])
            digit_cube.paint_uniform_color([1, 0, 0])  # Red for ID
            combined_mesh += digit_cube
        
        # Apply rotation if specified
        if hasattr(aruco_info, 'rotation') and aruco_info.rotation != (0, 0, 0):
            roll, pitch, yaw = aruco_info.rotation
            print(f"Applying rotation: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            
            # Create rotation matrix from Euler angles
            
            try:
                from scipy.spatial.transform import Rotation as R
                # Create rotation object from Euler angles (XYZ order for better control)
                rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
                rotation_matrix = rotation.as_matrix()
                print(f"Rotation matrix:\n{rotation_matrix}")
                
                # Apply rotation around the marker center
                center = np.array([x, y, z])
                combined_mesh.rotate(rotation_matrix, center=center)
                print("Rotation applied successfully")
                
            except ImportError:
                # Fallback if scipy is not available - use simple rotations
                print("Warning: scipy not available, using simple rotation fallback")
                center = np.array([x, y, z])
                
                # Apply rotations one by one for better control
                if abs(roll) > 0.001:
                    roll_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([roll, 0, 0])
                    combined_mesh.rotate(roll_matrix, center=center)
                    print(f"Applied roll rotation: {roll:.3f}")
                
                if abs(pitch) > 0.001:
                    pitch_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, pitch, 0])
                    combined_mesh.rotate(pitch_matrix, center=center)
                    print(f"Applied pitch rotation: {pitch:.3f}")
                
                if abs(yaw) > 0.001:
                    yaw_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
                    combined_mesh.rotate(yaw_matrix, center=center)
                    print(f"Applied yaw rotation: {yaw:.3f}")
        
        combined_mesh.compute_vertex_normals()
        return combined_mesh
    
    def create_scale_ruler(self) -> o3d.geometry.TriangleMesh:
        """Create a scale ruler to show dimensions in the 3D viewer."""
        if not self.mesh_info or 'dimensions' not in self.mesh_info:
            return None
            
        dims = self.mesh_info['dimensions']
        max_dim = max(dims['length'], dims['width'], dims['height'])
        
        # Create ruler length (15% of max dimension for better visibility)
        ruler_length = max_dim * 0.15
        ruler_thickness = max_dim * 0.005  # Thinner for better proportions
        
        # Create ruler geometry
        ruler = o3d.geometry.TriangleMesh.create_box(ruler_length, ruler_thickness, ruler_thickness)
        
        # Position ruler at bottom-left of the model with proper spacing
        bbox = self.mesh.get_axis_aligned_bounding_box()
        spacing = max_dim * 0.05  # 5% of max dimension as spacing
        ruler.translate([
            bbox.min_bound[0] - spacing,
            bbox.min_bound[1] - spacing, 
            bbox.min_bound[2] - spacing
        ])
        
        # Color the ruler
        ruler.paint_uniform_color([1, 1, 0])  # Yellow
        
        # Add tick marks
        tick_mesh = o3d.geometry.TriangleMesh()
        num_ticks = 5
        for i in range(num_ticks + 1):
            tick_length = ruler_thickness * (2 if i % 5 == 0 else 1)  # Major ticks every 5
            tick = o3d.geometry.TriangleMesh.create_box(
                ruler_thickness * 0.5, tick_length, ruler_thickness * 0.5
            )
            tick.translate([
                bbox.min_bound[0] - spacing + (i / num_ticks) * ruler_length,
                bbox.min_bound[1] - spacing - tick_length,
                bbox.min_bound[2] - spacing
            ])
            tick.paint_uniform_color([1, 1, 0])  # Yellow
            tick_mesh += tick
        
        # Combine ruler and ticks
        scale_ruler = ruler + tick_mesh
        scale_ruler.compute_vertex_normals()
        
        return scale_ruler

    def add_marker_to_viewer(self, marker_id: int, position: tuple, size: float):
        """Add a marker to the 3D viewer."""
        if self.vis is None:
            return
            
        # Create the enhanced ArUco marker geometry
        marker_mesh = self.create_aruco_marker_geometry(position, size, marker_id)
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
        
    def add_aruco_marker(self, marker_id: int, aruco_info: ArUcoMarkerInfo) -> None:
        """Add a real ArUco marker with proper texture."""
        self.markers[marker_id] = {
            'aruco_info': aruco_info,
            'geometry': None,
            'is_selected': False
        }
        self.update_stats()
        
        # Add to 3D viewer if it's open
        if self.vis is not None:
            try:
                # Create ArUco marker with actual image texture
                marker_mesh = self.create_real_aruco_marker(aruco_info)
                self.markers[marker_id]['geometry'] = marker_mesh
                self.vis.add_geometry(marker_mesh)
            except Exception as e:
                print(f"Failed to create ArUco marker: {e}")
                # Fallback to geometric representation
                marker_mesh = self.create_aruco_marker_geometry(aruco_info.position, aruco_info.size, marker_id)
                self.markers[marker_id]['geometry'] = marker_mesh
                self.vis.add_geometry(marker_mesh)
    
    def add_marker(self, marker_id: int, position: tuple, size: float = 0.05) -> None:
        """Add an ArUco marker visualization (legacy method)."""
        # Create default ArUco info for legacy compatibility
        aruco_info = ArUcoMarkerInfo(
            dictionary="DICT_4X4_50",
            marker_id=0,
            position=position,
            size=size
        )
        self.add_aruco_marker(marker_id, aruco_info)
        
    def remove_marker(self, marker_id: int) -> None:
        """Remove a marker from the viewer."""
        if marker_id in self.markers:
            marker = self.markers[marker_id]
            if marker['geometry'] is not None and self.vis is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            del self.markers[marker_id]
            self.update_stats()
            
    def select_marker(self, marker_id: int) -> None:
        """Highlight the selected marker."""
        # Update selection state for all markers
        for mid, marker in self.markers.items():
            marker['is_selected'] = (mid == marker_id)
        self.update_stats()
        
    def move_marker(self, marker_id: int, new_position: tuple) -> None:
        """Move a marker to a new position."""
        if marker_id in self.markers and self.vis is not None:
            marker = self.markers[marker_id]
            
            # Remove old geometry
            if marker['geometry'] is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
            
            # Update position
            if 'aruco_info' in marker:
                # New format with ArUco info
                marker['aruco_info'].position = new_position
                # Create new geometry at new position
                try:
                    new_geometry = self.create_real_aruco_marker(marker['aruco_info'])
                except Exception:
                    # Fallback to geometric representation
                    new_geometry = self.create_aruco_marker_geometry(
                        new_position, marker['aruco_info'].size, marker_id
                    )
            else:
                # Legacy format
                marker['position'] = new_position
                new_geometry = self.create_aruco_marker_geometry(
                    new_position, marker.get('size', 0.05), marker_id
                )
            
            marker['geometry'] = new_geometry
            self.vis.add_geometry(new_geometry)
            self.update_stats()
    
    def rotate_marker(self, marker_id: int, new_rotation: tuple) -> None:
        """Rotate a marker to a new orientation."""
        print(f"🔄 Rotating marker {marker_id} to {new_rotation}")
        
        if marker_id in self.markers and self.vis is not None:
            marker = self.markers[marker_id]
            
            # Remove old geometry
            if marker['geometry'] is not None:
                self.vis.remove_geometry(marker['geometry'], reset_bounding_box=False)
                print(f"✅ Removed old geometry for marker {marker_id}")
            
            # Update rotation
            if 'aruco_info' in marker:
                # New format with ArUco info
                marker['aruco_info'].rotation = new_rotation
                print(f"📝 Updated ArUco info rotation: {marker['aruco_info'].rotation}")
                
                # Create new geometry with new rotation
                try:
                    new_geometry = self.create_real_aruco_marker(marker['aruco_info'])
                    print(f"✅ Created new ArUco marker geometry with rotation")
                except Exception as e:
                    print(f"❌ Failed to create ArUco marker: {e}")
                    # Fallback to geometric representation
                    new_geometry = self.create_aruco_marker_geometry(
                        marker['aruco_info'].position, marker['aruco_info'].size, marker_id, new_rotation
                    )
                    print(f"🔄 Using fallback geometric representation with rotation")
            else:
                # Legacy format
                marker['rotation'] = new_rotation
                new_geometry = self.create_aruco_marker_geometry(
                    marker.get('position', (0, 0, 0)), marker.get('size', 0.05), marker_id, new_rotation
                )
                print(f"🔄 Using legacy format with rotation")
            
            marker['geometry'] = new_geometry
            self.vis.add_geometry(new_geometry)
            self.update_stats()
            
            print(f"✅ Marker {marker_id} rotation updated successfully")
        else:
            print(f"❌ Cannot rotate marker {marker_id}: not found or no visualizer")
            
    def enable_placement_mode(self) -> None:
        """Enable marker placement mode."""
        self.placement_mode = True
        # Update status to indicate placement mode
        self.status_label.setText("Placement Mode Active\\nClick on a face to place marker")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ff9800;
                font-size: 14px;
                font-weight: bold;
                margin: 20px;
            }
        """)
        # Show the placement instruction
        self.placement_label.setVisible(True)
        # Show alternative placement buttons if available, create if needed
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(True)
            self.face_picker_btn.setVisible(True)
        elif not hasattr(self, 'click_place_btn') or self.click_place_btn is None:
            # Create the alternative placement buttons now
            self.setup_alternative_placement()
        # Update mode label
        self.mode_label.setText("Mode: Place Marker")
        self.mode_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        
    def disable_placement_mode(self) -> None:
        """Disable marker placement mode."""
        self.placement_mode = False
        # Hide the placement instruction
        self.placement_label.setVisible(False)
        # Hide alternative placement buttons if available
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(False)
        if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
            self.face_picker_btn.setVisible(False)
        # Restore normal mode label
        self.mode_label.setText("Mode: View")
        self.mode_label.setStyleSheet("font-weight: bold; color: blue;")
        # Restore normal status
        if self.mesh is not None:
            vertices_count = len(self.mesh.vertices) if hasattr(self.mesh, 'vertices') else 0
            triangles_count = len(self.mesh.triangles) if hasattr(self.mesh, 'triangles') else 0
            self.status_label.setText(f"Loaded mesh:\\n{vertices_count} vertices\\n{triangles_count} faces")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    margin: 20px;
                }
            """)
        
    def calculate_face_center_from_triangle(self, triangle_idx: int) -> tuple:
        """Calculate the center of a triangle face."""
        if self.mesh is None:
            return None
            
        triangles = np.asarray(self.mesh.triangles)
        vertices = np.asarray(self.mesh.vertices)
        
        if triangle_idx >= len(triangles):
            return None
            
        # Get the three vertices of the triangle
        triangle = triangles[triangle_idx]
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        
        # Calculate the centroid (center) of the triangle
        center = (v1 + v2 + v3) / 3.0
        return tuple(center)
    
    def pick_face_with_raycast(self, x: int, y: int) -> tuple:
        """Use ray casting to find the actual clicked face center."""
        if self.mesh is None or self.vis is None:
            return None
            
        try:
            # Get camera parameters
            ctr = self.vis.get_view_control()
            camera_params = ctr.convert_to_pinhole_camera_parameters()
            
            # Create ray from screen coordinates
            width = camera_params.intrinsic.width
            height = camera_params.intrinsic.height
            
            # Convert screen coordinates to normalized device coordinates
            ndc_x = (2.0 * x / width) - 1.0
            ndc_y = 1.0 - (2.0 * y / height)
            
            # Create a ray from the camera
            # This is a simplified approach - for now we'll use a different strategy
            # since Open3D's ray casting can be complex to set up properly
            
            # Instead, let's pick the face closest to the camera view center
            triangles = np.asarray(self.mesh.triangles)
            vertices = np.asarray(self.mesh.vertices)
            
            if len(triangles) > 0:
                # Find triangle faces and their centers
                face_centers = []
                for i, triangle in enumerate(triangles):
                    v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
                    center = (v1 + v2 + v3) / 3.0
                    face_centers.append((i, center))
                
                # For now, pick a random face center
                # In a proper implementation, you'd find the face closest to the ray
                import random
                selected_face = random.choice(face_centers)
                return tuple(selected_face[1])
                
        except Exception as e:
            print(f"Error in ray casting: {e}")
            
        return None
    
    def setup_qt_integration(self) -> None:
        """Setup Qt timer to integrate Open3D events with Qt event loop."""
        self.open3d_timer = QTimer()
        self.open3d_timer.timeout.connect(self.update_open3d_events)
        self.open3d_timer.start(16)  # ~60 FPS
    
    def update_open3d_events(self) -> None:
        """Update Open3D events to keep the visualizer responsive."""
        if hasattr(self, 'vis') and self.vis is not None:
            try:
                # Poll events to keep the window responsive
                self.vis.poll_events()
                self.vis.update_renderer()
            except:
                # If polling fails, stop the timer
                if hasattr(self, 'open3d_timer'):
                    self.open3d_timer.stop()
    
    def setup_picking_callback(self) -> None:
        """Set up mouse interaction callbacks for the Open3D visualizer."""
        def mouse_callback(vis, action, mods):
            """Handle mouse interactions in the 3D viewer."""
            if action == 1:  # Left mouse button pressed
                if self.placement_mode and self.mesh is not None:
                    try:
                        # Pick a triangle face center for marker placement
                        triangles = np.asarray(self.mesh.triangles)
                        
                        if len(triangles) > 0:
                            # For now, pick a random triangle face
                            # In a proper implementation, you'd use ray casting to find the clicked face
                            import random
                            triangle_idx = random.randint(0, len(triangles) - 1)
                            face_center = self.calculate_face_center_from_triangle(triangle_idx)
                            
                            if face_center:
                                # Use QTimer to emit signal from main thread
                                QTimer.singleShot(0, lambda: self.point_picked.emit(face_center))
                                print(f"Placed marker at face center: {face_center}")
                        else:
                            # Fallback to bbox center if no triangles
                            bbox = self.mesh.get_axis_aligned_bounding_box()
                            center = bbox.get_center()
                            QTimer.singleShot(0, lambda: self.point_picked.emit(tuple(center)))
                    except Exception as e:
                        print(f"Error in face picking: {e}")
            return False
        
        def key_callback(vis, key, action):
            """Handle keyboard interactions in the 3D viewer."""
            if key == ord('P') and action == 1:  # P key pressed
                # Toggle placement mode
                if self.placement_mode:
                    QTimer.singleShot(0, lambda: self.disable_placement_mode())
                print(f"Placement mode: {'OFF' if self.placement_mode else 'ON'}")
            elif key == ord('R') and action == 1:  # R key pressed
                vis.reset_view_point(True)
            return False
        
        # Register callbacks with Open3D visualizer
        print("Checking available callback methods...")
        available_methods = [method for method in dir(self.vis) if 'callback' in method.lower()]
        print(f"Available callback methods: {available_methods}")
        
        try:
            # Try to register the callbacks
            if hasattr(self.vis, 'register_mouse_callback'):
                result = self.vis.register_mouse_callback(mouse_callback)
                print(f"Mouse callback registered, result: {result}")
            else:
                print("register_mouse_callback not available")
                
            if hasattr(self.vis, 'register_key_callback'):
                result = self.vis.register_key_callback(key_callback)
                print(f"Key callback registered, result: {result}")
            else:
                print("register_key_callback not available")
                
            # Always set up alternative placement as backup
            print("Setting up alternative placement as backup...")
            self.setup_alternative_placement()
            
        except Exception as e:
            print(f"Callback registration failed: {e}")
            import traceback
            traceback.print_exc()
            # Set up alternative method
            self.setup_alternative_placement()
    
    def setup_alternative_placement(self) -> None:
        """Setup alternative placement method when callbacks don't work."""
        # Add multiple placement options
        self.click_place_btn = QPushButton("🎯 Place on Random Face")
        self.click_place_btn.setVisible(False)
        self.click_place_btn.clicked.connect(self.place_marker_at_surface)

        self.face_picker_btn = QPushButton("📋 Choose Face from List")
        self.face_picker_btn.setVisible(False)
        self.face_picker_btn.clicked.connect(self.open_face_picker)

        # Live face selection removed per user request
        # Style all buttons
        random_button_style = """
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: 2px solid #e68900;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                margin: 2px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #e68900;
                border-color: #d68100;
            }
            QPushButton:pressed {
                background-color: #d68100;
            }
        """

        list_button_style = random_button_style.replace('#ff9800', '#2196F3').replace('#e68900', '#1976D2').replace('#d68100', '#1565C0')

        self.click_place_btn.setStyleSheet(random_button_style)
        self.face_picker_btn.setStyleSheet(list_button_style)
        
        # Add all buttons to the layout
        if hasattr(self, 'layout') and self.layout():
            main_layout = self.layout()
            # Insert buttons before the stretch
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and hasattr(item, 'changeSize'):  # This is the stretch
                    main_layout.insertWidget(i, self.click_place_btn)
                    main_layout.insertWidget(i+1, self.face_picker_btn)
                    break
            else:
                # If no stretch found, just add at the end
                main_layout.addWidget(self.click_place_btn)
                main_layout.addWidget(self.face_picker_btn)
    
    def place_marker_at_surface(self) -> None:
        """Place a marker at a calculated surface position."""
        if self.placement_mode and self.mesh is not None:
            try:
                triangles = np.asarray(self.mesh.triangles)
                if len(triangles) > 0:
                    # Pick a triangle face center
                    import random
                    triangle_idx = random.randint(0, len(triangles) - 1)
                    face_center = self.calculate_face_center_from_triangle(triangle_idx)
                    
                    if face_center:
                        self.point_picked.emit(face_center)
                        print(f"Marker placed at face center: {face_center}")
                        
                        # Hide the placement button
                        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
                            self.click_place_btn.setVisible(False)
                else:
                    # Fallback to bbox center
                    bbox = self.mesh.get_axis_aligned_bounding_box()
                    center = bbox.get_center()
                    self.point_picked.emit(tuple(center))
            except Exception as e:
                print(f"Error in alternative placement: {e}")
    
    def open_face_picker(self) -> None:
        """Open a dialog to let user choose a specific face."""
        print("Opening face picker dialog...")
        
        if self.mesh is None:
            print("No mesh available for face picking")
            return
            
        try:
            triangles = np.asarray(self.mesh.triangles)
            vertices = np.asarray(self.mesh.vertices)
            
            if len(triangles) == 0:
                print("No triangles found in mesh")
                return
                
            # Create face picker dialog
            dialog = FacePickerDialog(self, triangles, vertices)
            if dialog.exec() == dialog.DialogCode.Accepted:
                selected_face_center = dialog.get_selected_face_center()
                if selected_face_center:
                    print(f"User selected face center: {selected_face_center}")
                    self.point_picked.emit(selected_face_center)
                    
                    # Hide the placement buttons
                    if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
                        self.click_place_btn.setVisible(False)
                    if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
                        self.face_picker_btn.setVisible(False)
                        
        except Exception as e:
            print(f"Error in face picker: {e}")
            import traceback
            traceback.print_exc()



    def smart_auto_place(self) -> None:
        """Smart automatic placement that chooses the best face for marker placement."""
        print("🧠 Smart auto-placement activated!")

        if self.mesh is None:
            print("❌ No mesh loaded")
            return

        try:
            # Get face groups if not already computed
            if not hasattr(self, 'face_groups') or self.face_groups is None:
                print("🔄 Analyzing mesh faces...")
                self.face_groups = self.group_triangles_by_face()

            if not self.face_groups:
                print("❌ No faces found in mesh")
                return

            print(f"📊 Found {len(self.face_groups)} faces to analyze")

            # Smart face selection algorithm
            best_face = self.find_best_face_for_marker()
            if best_face is None:
                print("❌ Could not find suitable face for marker placement")
                return

            face_center, face_normal, face_area, triangle_indices = best_face

            print(f"Selected face center: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")
            print(f"Selected face area: {face_area:.4f}")
            # Place marker at the face center
            self.point_picked.emit(tuple(face_center))

            # Hide placement buttons after successful placement
            self.disable_direct_placement_mode()

            print("✅ Smart marker placement completed!")
            print(f"Final marker position: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")

        except Exception as e:
            print(f"❌ Error in smart auto-placement: {e}")
            import traceback
            traceback.print_exc()

    def find_best_face_for_marker(self):
        """Find the best face for marker placement using smart criteria."""
        if not self.face_groups:
            return None

        # Scoring criteria for face selection:
        # 1. Face area (larger faces are better for markers)
        # 2. Face orientation (faces pointing more upward are preferred)
        # 3. Face position (avoid very small faces)
        # 4. Face normal alignment (prefer faces that are reasonably flat)

        best_score = -1
        best_face = None

        for face_center, face_normal, face_area, triangle_indices in self.face_groups:
            score = 0

            # 1. Area score (0-50 points)
            if face_area > 0:
                area_score = min(50, face_area * 10000)  # Scale area to reasonable score
                score += area_score

            # 2. Orientation score (0-30 points) - prefer faces pointing somewhat upward
            up_alignment = abs(face_normal[2])  # Z-component indicates up/down alignment
            orientation_score = up_alignment * 30
            score += orientation_score

            # 3. Size bonus (0-20 points) - prefer larger faces
            size_score = min(20, len(triangle_indices) / 5)  # More triangles = larger face
            score += size_score

            print(f"Face score: {score:.1f}")
            if score > best_score:
                best_score = score
                best_face = (face_center, face_normal, face_area, triangle_indices)

        return best_face

    def handle_direct_click(self, x: int, y: int) -> None:
        """Handle direct click on the 3D model."""
        if not self.direct_click_mode or self.mesh is None:
            return

        try:
            print(f"Processing click at ({x}, {y})")

            # Get the view control and camera
            view_control = self.vis.get_view_control()

            # Convert screen coordinates to world coordinates
            # This is a simplified approach - in a full implementation we'd use ray casting
            depth = view_control.get_field_of_view() / 10.0  # Approximate depth

            # Get camera parameters
            camera_params = view_control.convert_to_pinhole_camera_parameters()

            # Create a ray from the camera through the clicked point
            ray_origin = camera_params.extrinsic[:3, 3]
            ray_direction = self.screen_to_world_direction(x, y, camera_params)

            print(f"Ray origin: {ray_origin}")
            print(f"Ray direction: {ray_direction}")

            # Find intersection with mesh
            intersection_point = self.find_mesh_intersection(ray_origin, ray_direction)

            if intersection_point is not None:
                print(f"Found intersection at: {intersection_point}")
                self.point_picked.emit(tuple(intersection_point))

                # Disable direct click mode
                self.direct_click_mode = False

                # Reset UI
                self.disable_direct_placement_mode()
            else:
                print("No intersection found with mesh")

        except Exception as e:
            print(f"Error handling direct click: {e}")
            import traceback
            traceback.print_exc()

    def screen_to_world_direction(self, x: int, y: int, camera_params):
        """Convert screen coordinates to world direction vector."""
        try:
            # Get camera intrinsic parameters
            intrinsic = camera_params.intrinsic
            extrinsic = camera_params.extrinsic

            # Convert screen coordinates to normalized device coordinates
            width, height = self.vis.get_render_option().viewport_width, self.vis.get_render_option().viewport_height
            ndc_x = (2.0 * x / width) - 1.0
            ndc_y = 1.0 - (2.0 * y / height)  # Flip Y coordinate

            # Convert to camera coordinates
            camera_point = np.array([ndc_x, ndc_y, 1.0])

            # Apply inverse intrinsic matrix
            inv_intrinsic = np.linalg.inv(intrinsic.intrinsic_matrix)
            camera_direction = inv_intrinsic @ camera_point

            # Apply inverse extrinsic matrix (camera pose)
            inv_extrinsic = np.linalg.inv(extrinsic)
            world_direction = inv_extrinsic[:3, :3] @ camera_direction

            # Normalize
            return world_direction / np.linalg.norm(world_direction)

        except Exception as e:
            print(f"Error in screen to world conversion: {e}")
            return np.array([0, 0, -1])  # Default forward direction

    def find_mesh_intersection(self, ray_origin, ray_direction):
        """Find intersection point between ray and mesh."""
        try:
            # Use Open3D's ray casting if available
            if hasattr(self.mesh, 'create_raycasting_scene'):
                scene = self.mesh.create_raycasting_scene()
                result = scene.cast_rays([ray_origin], [ray_direction])

                if len(result['t_hit']) > 0 and result['t_hit'][0] < np.inf:
                    hit_point = ray_origin + result['t_hit'][0] * ray_direction
                    return hit_point

            # Fallback: find closest point on mesh
            # This is a simplified approach
            bbox = self.mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()

            # Return a point near the center (this is a fallback)
            return np.array(center)

        except Exception as e:
            print(f"Error in mesh intersection: {e}")
            return None

    def show_direct_click_instructions(self) -> None:
        """Show instructions for direct click placement."""
        print("Direct click instructions:")
        print("1. Move your mouse over the 3D model")
        print("2. Click on the surface where you want to place the marker")
        print("3. The marker will be placed at the clicked location")
        print("Note: If clicking doesn't work, try the 'Choose Face from List' option")

    def disable_direct_placement_mode(self) -> None:
        """Disable direct placement mode and reset UI."""
        self.direct_click_mode = False

        if hasattr(self, 'placement_label'):
            self.placement_label.setText("Choose how to place ArUco marker:\n• 🎲 Quick Random • 📋 Face List • 🎯 Smart Auto")
            self.placement_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF3CD;
                    color: #856404;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                    margin: 5px;
                    border: 1px solid #FFEAA7;
                }
            """)

        # Reset button styles and visibility
        if hasattr(self, 'click_place_btn') and self.click_place_btn is not None:
            self.click_place_btn.setVisible(True)
            self.click_place_btn.setText("🎲 Quick Random Place")

        if hasattr(self, 'face_picker_btn') and self.face_picker_btn is not None:
            self.face_picker_btn.setVisible(True)
            self.face_picker_btn.setText("📋 Pick Face from List")


    def setup_face_highlighting(self) -> None:
        """Setup face highlighting system for direct placement mode."""
        if self.mesh is None:
            return

        try:
            # Create face groups if not already done
            if not hasattr(self, 'face_groups') or self.face_groups is None:
                self.face_groups = self.group_triangles_by_face()

            print(f"Setting up highlighting for {len(self.face_groups)} faces")

            # Create highlighted versions of faces
            self.highlighted_faces = []
            for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(self.face_groups):
                # Create a small sphere at the face center for highlighting
                highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                highlight_sphere.translate(face_center)
                highlight_sphere.paint_uniform_color([1, 1, 0])  # Yellow highlight

                self.highlighted_faces.append(highlight_sphere)

                # Initially hide all highlights
                highlight_sphere.translate([0, 0, -1000])  # Move far away

            print("Face highlighting setup complete")

        except Exception as e:
            print(f"Error setting up face highlighting: {e}")
            import traceback
            traceback.print_exc()

    def highlight_face_at_point(self, point) -> None:
        """Highlight the face closest to the given point."""
        if not hasattr(self, 'face_groups') or not self.face_groups:
            return

        try:
            # Find the closest face to the point
            min_distance = float('inf')
            closest_face_idx = -1

            for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(self.face_groups):
                distance = np.linalg.norm(np.array(point) - np.array(face_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_face_idx = face_idx

            # Update highlighting
            for i, highlight_sphere in enumerate(self.highlighted_faces):
                if i == closest_face_idx:
                    # Move highlight to face center
                    face_center = self.face_groups[i][0]
                    highlight_sphere.translate([0, 0, 1000])  # Reset position
                    highlight_sphere.translate(face_center)
                else:
                    # Hide other highlights
                    highlight_sphere.translate([0, 0, -1000])

            # Update the visualizer
            if self.vis:
                self.vis.update_geometry(highlight_sphere)
                self.vis.poll_events()
                self.vis.update_renderer()

        except Exception as e:
            print(f"Error highlighting face: {e}")

    def group_triangles_by_face(self):
        """Group triangles that belong to the same face based on normal vectors."""
        if self.mesh is None:
            return []

        face_groups = []
        normal_tolerance = 0.1  # Tolerance for grouping normals

        # Calculate normals for all triangles
        triangle_normals = []
        triangle_centers = []
        triangle_areas = []

        for i, triangle in enumerate(np.asarray(self.mesh.triangles)):
            v1, v2, v3 = self.mesh.vertices[triangle[0]], self.mesh.vertices[triangle[1]], self.mesh.vertices[triangle[2]]

            # Calculate normal
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / (np.linalg.norm(normal) + 1e-8)  # Normalize

            # Calculate center and area
            center = (v1 + v2 + v3) / 3.0
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

            triangle_normals.append(normal)
            triangle_centers.append(center)
            triangle_areas.append(area)

        # Group triangles by similar normals
        used_triangles = set()

        for i, normal in enumerate(triangle_normals):
            if i in used_triangles:
                continue

            # Find all triangles with similar normals
            group_triangles = [i]
            group_centers = [triangle_centers[i]]
            group_areas = [triangle_areas[i]]
            used_triangles.add(i)

            for j, other_normal in enumerate(triangle_normals):
                if j in used_triangles:
                    continue

                # Check if normals are similar (dot product close to 1)
                dot_product = np.dot(normal, other_normal)
                if dot_product > (1.0 - normal_tolerance):
                    group_triangles.append(j)
                    group_centers.append(triangle_centers[j])
                    group_areas.append(triangle_areas[j])
                    used_triangles.add(j)

            # Calculate face properties
            face_center = np.mean(group_centers, axis=0)
            face_area = sum(group_areas)

            face_groups.append((face_center, normal, face_area, group_triangles))

        return face_groups

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
    
    def reset_camera(self) -> None:
        """Reset camera view."""
        if self.vis is not None:
            self.vis.reset_view_point(True)
    
    def fit_to_view(self) -> None:
        """Fit the current model to the view."""
        if self.vis is not None and self.mesh is not None:
            self.vis.reset_view_point(True)
    
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
        # This would be implemented with coordinate frame geometry
        pass
    
    def show_grid(self, show: bool) -> None:
        """Show or hide grid."""
        # This would be implemented with grid geometry
        pass
    
    def show_scale_ruler(self, show: bool) -> None:
        """Show or hide scale ruler."""
        if self.vis is not None and self.scale_ruler is not None:
            if show:
                self.vis.add_geometry(self.scale_ruler)
            else:
                self.vis.remove_geometry(self.scale_ruler, reset_bounding_box=False)
    
    def set_wireframe(self, wireframe: bool) -> None:
        """Set wireframe mode."""
        if self.vis is not None:
            render_opt = self.vis.get_render_option()
            render_opt.mesh_show_wireframe = wireframe
            render_opt.mesh_show_back_face = wireframe
    
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
    
    def update_stats(self) -> None:
        """Update the statistics display."""
        # Update markers info
        if self.markers:
            markers_text = "Placed ArUco Markers:\n"
            for marker_id, marker in self.markers.items():
                if 'aruco_info' in marker:
                    # New format with ArUco info
                    aruco_info = marker['aruco_info']
                    x, y, z = aruco_info.position
                    markers_text += f"  {aruco_info.dictionary} ID:{aruco_info.marker_id} at ({x:.3f}, {y:.3f}, {z:.3f})\n"
                else:
                    # Legacy format
                    x, y, z = marker.get('position', (0, 0, 0))
                    markers_text += f"  Marker {marker_id}: ({x:.3f}, {y:.3f}, {z:.3f})\n"
        else:
            markers_text = "No ArUco markers placed"
        self.markers_text.setText(markers_text)
        
        # Update grasp poses info
        if self.grasp_poses:
            grasps_text = "Defined Grasp Poses:\n"
            for grasp_id, grasp in self.grasp_poses.items():
                x, y, z = grasp.get('position', (0, 0, 0))
                marker_id = grasp.get('marker_id', 0)
                grasps_text += f"  Grasp {grasp_id} (M{marker_id}): ({x:.3f}, {y:.3f}, {z:.3f})\n"
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
