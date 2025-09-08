"""Face picker dialog for selecting specific faces from a mesh."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QApplication
)


class FacePickerDialog(QWidget):
    """Dialog for picking a specific face from the mesh."""
    
    def __init__(self, parent, triangles, vertices):
        super().__init__()
        self.parent = parent
        self.triangles = triangles
        self.vertices = vertices
        self.selected_face_data = None
        self.dialog_result = 0
        
        self.setWindowTitle("Select Face for ArUco Marker Placement")
        self.setGeometry(200, 200, 600, 500)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Choose a Face for ArUco Marker Placement")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("Select a face from the list below. The ArUco marker will be placed at the center of the selected face.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(instructions)
        
        # Face list
        self.face_list = QListWidget()
        self.face_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                font-family: monospace;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
        """)
        
        # Populate face list with face information
        self.populate_face_list()
        layout.addWidget(self.face_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #616161; }
        """)
        
        self.select_btn = QPushButton("🎯 Place Marker Here")
        self.select_btn.clicked.connect(self.accept)
        self.select_btn.setEnabled(False)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.select_btn)
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.face_list.itemSelectionChanged.connect(self.on_selection_changed)
        
    def populate_face_list(self):
        """Populate the face list with actual faces (not individual triangles)."""
        print(f"Analyzing {len(self.triangles)} triangles to find actual faces...")
        print("NOTE: Algorithm finds faces DYNAMICALLY - it will find as many faces as exist in your model!")

        # Group triangles by their normal vectors to find actual faces
        face_groups = self.group_triangles_by_face()

        print(f"Algorithm found {len(face_groups)} actual faces (not hardcoded!)")
        print("Each face is made up of triangles pointing in the same direction.")
        
        for face_idx, (face_center, face_normal, face_area, triangle_indices) in enumerate(face_groups):
            # Determine face orientation
            face_name = self.get_face_name(face_normal)
            
            item_text = f"{face_name}: Center ({face_center[0]:6.3f}, {face_center[1]:6.3f}, {face_center[2]:6.3f}) | Area: {face_area:.4f}"
            
            item = QListWidgetItem(item_text)
            # Store the face center and representative triangle index
            item.setData(Qt.ItemDataRole.UserRole, {
                'face_center': face_center,
                'triangle_idx': triangle_indices[0],  # Use first triangle as representative
                'face_normal': face_normal,
                'area': face_area
            })
            self.face_list.addItem(item)
            
        print(f"Added {self.face_list.count()} faces to the list")
        
    def group_triangles_by_face(self):
        """Group triangles that belong to the same face based on normal vectors."""
        face_groups = []
        normal_tolerance = 0.1  # Tolerance for grouping normals
        
        # Calculate normals for all triangles
        triangle_normals = []
        triangle_centers = []
        triangle_areas = []
        
        for i, triangle in enumerate(self.triangles):
            v1, v2, v3 = self.vertices[triangle[0]], self.vertices[triangle[1]], self.vertices[triangle[2]]
            
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

        print(f"Starting face detection algorithm...")
        print(f"Total triangles to process: {len(triangle_normals)}")
        print(f"Algorithm will find ALL faces based on normal vectors - NO hardcoded limit!")

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

            print(f"Face {len(face_groups)+1}: {len(group_triangles)} triangles, center=({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f}), area={face_area:.4f}")

            face_groups.append((face_center, normal, face_area, group_triangles))
        
        return face_groups
    
    def get_face_name(self, normal):
        """Get a descriptive name for the face based on its normal vector."""
        # Normalize the normal vector
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Define primary directions
        directions = {
            'Top': np.array([0, 0, 1]),
            'Bottom': np.array([0, 0, -1]),
            'Front': np.array([0, 1, 0]),
            'Back': np.array([0, -1, 0]),
            'Right': np.array([1, 0, 0]),
            'Left': np.array([-1, 0, 0])
        }
        
        # Find the closest direction
        best_match = 'Unknown'
        best_similarity = -1
        
        for name, direction in directions.items():
            similarity = abs(np.dot(normal, direction))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return f"{best_match} Face"
        
    def on_selection_changed(self):
        """Handle face selection change."""
        current_item = self.face_list.currentItem()
        if current_item:
            self.selected_face_data = current_item.data(Qt.ItemDataRole.UserRole)
            self.select_btn.setEnabled(True)
            face_center = self.selected_face_data['face_center']
            print(f"Selected face center: ({face_center[0]:.3f}, {face_center[1]:.3f}, {face_center[2]:.3f})")
        else:
            self.selected_face_data = None
            self.select_btn.setEnabled(False)
            
    def get_selected_face_center(self):
        """Get the center of the selected face."""
        if self.selected_face_data is not None:
            return tuple(self.selected_face_data['face_center'])
        return None
        
    def accept(self):
        """Accept the selection and close."""
        self.dialog_result = 1
        self.close()
        
    def reject(self):
        """Reject the selection and close."""
        self.dialog_result = 0
        self.close()
        
    def exec(self):
        """Show the dialog modally."""
        self.show()
        # Process events until dialog is closed
        while self.isVisible():
            QApplication.processEvents()
        return self.dialog_result
        
    class DialogCode:
        Accepted = 1
        Rejected = 0
