"""Panel for managing grasp poses on the 3D model."""

from typing import List, Dict, Any, Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel,
    QSpinBox, QDoubleSpinBox, QInputDialog, QMessageBox,
    QComboBox, QLineEdit
)


class GraspListItem(QListWidgetItem):
    """Custom list item for grasp poses."""
    
    def __init__(self, grasp_id: int, name: str, marker_id: int, position: tuple, orientation: tuple):
        super().__init__()
        self.grasp_id = grasp_id
        self.name = name
        self.marker_id = marker_id
        self.position = position
        self.orientation = orientation  # quaternion (w, x, y, z)
        self.update_text()
        
    def update_text(self) -> None:
        """Update the display text."""
        self.setText(f"{self.name} (M{self.marker_id})")
        
    def get_data(self) -> Dict[str, Any]:
        """Get grasp data as dictionary."""
        return {
            "id": self.grasp_id,
            "name": self.name,
            "marker_id": self.marker_id,
            "position": list(self.position),
            "orientation": list(self.orientation)
        }


class GraspPanel(QGroupBox):
    """Panel for managing grasp poses."""
    
    # Signals
    grasp_added = pyqtSignal(int, int, tuple, tuple)    # grasp_id, marker_id, position, orientation
    grasp_removed = pyqtSignal(int)                     # grasp_id
    grasp_selected = pyqtSignal(int)                    # grasp_id
    grasp_modified = pyqtSignal(int, tuple, tuple)      # grasp_id, position, orientation
    
    def __init__(self):
        super().__init__("Grasp Poses")
        self.next_grasp_id = 0
        self.grasps: Dict[int, GraspListItem] = {}
        self.available_markers: List[int] = []
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Add grasp section
        add_layout = QVBoxLayout()
        
        # Grasp name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter grasp name...")
        name_layout.addWidget(self.name_input)
        add_layout.addLayout(name_layout)
        
        # Marker selection
        marker_layout = QHBoxLayout()
        marker_layout.addWidget(QLabel("Marker:"))
        self.marker_combo = QComboBox()
        self.marker_combo.setToolTip("Select the ArUco marker to attach this grasp to")
        marker_layout.addWidget(self.marker_combo)
        add_layout.addLayout(marker_layout)
        
        # Add button
        self.add_button = QPushButton("Add Grasp Pose")
        self.add_button.setToolTip("Click to place a new grasp pose")
        self.add_button.setEnabled(False)  # Disabled until marker is available
        add_layout.addWidget(self.add_button)
        
        layout.addLayout(add_layout)
        
        # Grasp list
        list_label = QLabel("Defined Grasp Poses:")
        layout.addWidget(list_label)
        
        self.grasp_list = QListWidget()
        self.grasp_list.setMaximumHeight(150)
        self.grasp_list.setToolTip("List of defined grasp poses")
        layout.addWidget(self.grasp_list)
        
        # Grasp controls
        controls_layout = QHBoxLayout()
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.setEnabled(False)
        self.edit_button.setToolTip("Edit selected grasp pose")
        controls_layout.addWidget(self.edit_button)
        
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setEnabled(False)
        self.duplicate_button.setToolTip("Duplicate selected grasp pose")
        controls_layout.addWidget(self.duplicate_button)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip("Remove selected grasp pose")
        controls_layout.addWidget(self.remove_button)
        
        layout.addLayout(controls_layout)
        
        # Position and orientation input
        transform_group = QGroupBox("Transform (relative to marker)")
        transform_layout = QVBoxLayout(transform_group)
        
        # Position
        pos_label = QLabel("Position (m):")
        transform_layout.addWidget(pos_label)
        
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("X:"))
        self.pos_x_spinbox = QDoubleSpinBox()
        self.pos_x_spinbox.setRange(-1.0, 1.0)
        self.pos_x_spinbox.setDecimals(3)
        pos_layout.addWidget(self.pos_x_spinbox)
        
        pos_layout.addWidget(QLabel("Y:"))
        self.pos_y_spinbox = QDoubleSpinBox()
        self.pos_y_spinbox.setRange(-1.0, 1.0)
        self.pos_y_spinbox.setDecimals(3)
        pos_layout.addWidget(self.pos_y_spinbox)
        
        pos_layout.addWidget(QLabel("Z:"))
        self.pos_z_spinbox = QDoubleSpinBox()
        self.pos_z_spinbox.setRange(-1.0, 1.0)
        self.pos_z_spinbox.setDecimals(3)
        pos_layout.addWidget(self.pos_z_spinbox)
        
        transform_layout.addLayout(pos_layout)
        
        # Orientation (Euler angles for simplicity)
        ori_label = QLabel("Orientation (degrees):")
        transform_layout.addWidget(ori_label)
        
        ori_layout = QHBoxLayout()
        ori_layout.addWidget(QLabel("Roll:"))
        self.roll_spinbox = QDoubleSpinBox()
        self.roll_spinbox.setRange(-180.0, 180.0)
        self.roll_spinbox.setDecimals(1)
        ori_layout.addWidget(self.roll_spinbox)
        
        ori_layout.addWidget(QLabel("Pitch:"))
        self.pitch_spinbox = QDoubleSpinBox()
        self.pitch_spinbox.setRange(-180.0, 180.0)
        self.pitch_spinbox.setDecimals(1)
        ori_layout.addWidget(self.pitch_spinbox)
        
        ori_layout.addWidget(QLabel("Yaw:"))
        self.yaw_spinbox = QDoubleSpinBox()
        self.yaw_spinbox.setRange(-180.0, 180.0)
        self.yaw_spinbox.setDecimals(1)
        ori_layout.addWidget(self.yaw_spinbox)
        
        transform_layout.addLayout(ori_layout)
        
        # Apply changes button
        self.apply_transform_button = QPushButton("Apply Transform")
        self.apply_transform_button.setEnabled(False)
        self.apply_transform_button.setToolTip("Apply transform to selected grasp pose")
        transform_layout.addWidget(self.apply_transform_button)
        
        layout.addWidget(transform_group)
        
        # Grasp type selection
        type_group = QGroupBox("Grasp Type")
        type_layout = QVBoxLayout(type_group)
        
        self.grasp_type_combo = QComboBox()
        self.grasp_type_combo.addItems([
            "Top Down", "Side Grasp", "Pinch Grasp", 
            "Power Grasp", "Precision Grasp", "Custom"
        ])
        type_layout.addWidget(self.grasp_type_combo)
        
        layout.addWidget(type_group)
        
        # Statistics
        self.stats_label = QLabel("Grasp Poses: 0")
        self.stats_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.stats_label)
        
    def setup_connections(self) -> None:
        """Setup signal-slot connections."""
        self.add_button.clicked.connect(self.add_grasp_pose)
        self.edit_button.clicked.connect(self.edit_selected_grasp)
        self.duplicate_button.clicked.connect(self.duplicate_selected_grasp)
        self.remove_button.clicked.connect(self.remove_selected_grasp)
        self.apply_transform_button.clicked.connect(self.apply_transform)
        
        self.grasp_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.grasp_list.itemDoubleClicked.connect(self.edit_selected_grasp)
        
        self.name_input.textChanged.connect(self.validate_inputs)
        self.marker_combo.currentTextChanged.connect(self.validate_inputs)
        
    def validate_inputs(self) -> None:
        """Validate input fields and enable/disable buttons."""
        has_name = len(self.name_input.text().strip()) > 0
        has_marker = self.marker_combo.count() > 0 and self.marker_combo.currentText() != ""
        
        self.add_button.setEnabled(has_name and has_marker)
        
    def add_grasp_pose(self) -> None:
        """Add a new grasp pose."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a name for the grasp pose.")
            return
            
        marker_text = self.marker_combo.currentText()
        if not marker_text:
            QMessageBox.warning(self, "No Marker", "Please select a marker first.")
            return
            
        try:
            marker_id = int(marker_text.split()[1])  # Extract ID from "Marker X"
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Invalid Marker", "Invalid marker selection.")
            return
            
        # Get position and orientation
        position = (
            self.pos_x_spinbox.value(),
            self.pos_y_spinbox.value(),
            self.pos_z_spinbox.value()
        )
        
        # Convert Euler angles to quaternion (simplified)
        orientation = self.euler_to_quaternion(
            self.roll_spinbox.value(),
            self.pitch_spinbox.value(),
            self.yaw_spinbox.value()
        )
        
        # Create grasp
        grasp_id = self.next_grasp_id
        item = GraspListItem(grasp_id, name, marker_id, position, orientation)
        self.grasps[grasp_id] = item
        
        # Add to list
        self.grasp_list.addItem(item)
        
        # Emit signal
        self.grasp_added.emit(grasp_id, marker_id, position, orientation)
        
        # Clear input
        self.name_input.clear()
        
        # Update stats
        self.update_stats()
        
        # Increment ID
        self.next_grasp_id += 1
        
    def remove_selected_grasp(self) -> None:
        """Remove the currently selected grasp pose."""
        current_item = self.grasp_list.currentItem()
        if current_item and isinstance(current_item, GraspListItem):
            reply = QMessageBox.question(
                self,
                "Remove Grasp Pose",
                f"Remove grasp pose '{current_item.name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove from list
                row = self.grasp_list.row(current_item)
                self.grasp_list.takeItem(row)
                
                # Remove from dictionary
                grasp_id = current_item.grasp_id
                del self.grasps[grasp_id]
                
                # Emit signal
                self.grasp_removed.emit(grasp_id)
                
                # Update stats
                self.update_stats()
                
    def edit_selected_grasp(self) -> None:
        """Edit the currently selected grasp pose."""
        current_item = self.grasp_list.currentItem()
        if current_item and isinstance(current_item, GraspListItem):
            # Get new name
            new_name, ok = QInputDialog.getText(
                self, "Edit Grasp Name", "Grasp name:", text=current_item.name
            )
            
            if ok and new_name.strip():
                current_item.name = new_name.strip()
                current_item.update_text()
                
    def duplicate_selected_grasp(self) -> None:
        """Duplicate the currently selected grasp pose."""
        current_item = self.grasp_list.currentItem()
        if current_item and isinstance(current_item, GraspListItem):
            # Create new grasp with same properties
            new_name = f"{current_item.name}_copy"
            grasp_id = self.next_grasp_id
            
            item = GraspListItem(
                grasp_id, new_name, current_item.marker_id, 
                current_item.position, current_item.orientation
            )
            self.grasps[grasp_id] = item
            
            # Add to list
            self.grasp_list.addItem(item)
            
            # Emit signal
            self.grasp_added.emit(
                grasp_id, current_item.marker_id, 
                current_item.position, current_item.orientation
            )
            
            # Update stats and increment ID
            self.update_stats()
            self.next_grasp_id += 1
            
    def apply_transform(self) -> None:
        """Apply transform changes to selected grasp pose."""
        current_item = self.grasp_list.currentItem()
        if current_item and isinstance(current_item, GraspListItem):
            # Get new transform
            position = (
                self.pos_x_spinbox.value(),
                self.pos_y_spinbox.value(),
                self.pos_z_spinbox.value()
            )
            
            orientation = self.euler_to_quaternion(
                self.roll_spinbox.value(),
                self.pitch_spinbox.value(),
                self.yaw_spinbox.value()
            )
            
            # Update item
            current_item.position = position
            current_item.orientation = orientation
            
            # Emit signal
            self.grasp_modified.emit(current_item.grasp_id, position, orientation)
            
    def on_selection_changed(self) -> None:
        """Handle grasp selection changes."""
        current_item = self.grasp_list.currentItem()
        has_selection = current_item is not None
        
        self.edit_button.setEnabled(has_selection)
        self.duplicate_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        self.apply_transform_button.setEnabled(has_selection)
        
        if has_selection and isinstance(current_item, GraspListItem):
            # Emit selection signal
            self.grasp_selected.emit(current_item.grasp_id)
            
            # Update transform controls
            x, y, z = current_item.position
            self.pos_x_spinbox.setValue(x)
            self.pos_y_spinbox.setValue(y)
            self.pos_z_spinbox.setValue(z)
            
            # Convert quaternion to Euler (simplified)
            roll, pitch, yaw = self.quaternion_to_euler(current_item.orientation)
            self.roll_spinbox.setValue(roll)
            self.pitch_spinbox.setValue(pitch)
            self.yaw_spinbox.setValue(yaw)
            
    def update_available_markers(self, marker_ids: List[int]) -> None:
        """Update the list of available markers."""
        self.available_markers = marker_ids
        self.marker_combo.clear()
        
        for marker_id in marker_ids:
            self.marker_combo.addItem(f"Marker {marker_id}")
            
        self.validate_inputs()
        
    def update_stats(self) -> None:
        """Update statistics display."""
        count = len(self.grasps)
        self.stats_label.setText(f"Grasp Poses: {count}")
        
    def get_all_grasp_poses(self) -> List[Dict[str, Any]]:
        """Get all grasp poses as a list of dictionaries."""
        return [item.get_data() for item in self.grasps.values()]
        
    def load_grasp_poses(self, grasps_data: List[Dict[str, Any]]) -> None:
        """Load grasp poses from data."""
        # Clear existing grasps
        self.grasp_list.clear()
        self.grasps.clear()
        
        # Load each grasp
        for grasp_data in grasps_data:
            grasp_id = grasp_data["id"]
            name = grasp_data["name"]
            marker_id = grasp_data["marker_id"]
            position = tuple(grasp_data["position"])
            orientation = tuple(grasp_data["orientation"])
            
            # Create item
            item = GraspListItem(grasp_id, name, marker_id, position, orientation)
            self.grasps[grasp_id] = item
            self.grasp_list.addItem(item)
            
            # Update next ID
            self.next_grasp_id = max(self.next_grasp_id, grasp_id + 1)
            
        # Update stats
        self.update_stats()
        
    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> tuple:
        """Convert Euler angles (degrees) to quaternion (w, x, y, z)."""
        # Simplified conversion - in practice, use scipy.spatial.transform.Rotation
        import math
        
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)
        
    def quaternion_to_euler(self, quaternion: tuple) -> tuple:
        """Convert quaternion (w, x, y, z) to Euler angles (degrees)."""
        # Simplified conversion - in practice, use scipy.spatial.transform.Rotation
        import math
        
        w, x, y, z = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
