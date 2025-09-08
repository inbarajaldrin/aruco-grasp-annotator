"""Panel for managing ArUco markers on the 3D model."""

from typing import List, Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel,
    QSpinBox, QDoubleSpinBox, QInputDialog, QMessageBox,
    QComboBox
)

from ..utils.aruco_utils import ArUcoGenerator, ArUcoMarkerInfo


class MarkerListItem(QListWidgetItem):
    """Custom list item for ArUco markers."""
    
    def __init__(self, marker_id: int, aruco_info: ArUcoMarkerInfo):
        super().__init__()
        self.marker_id = marker_id
        self.aruco_info = aruco_info
        self.update_text()
        
    def update_text(self) -> None:
        """Update the display text."""
        x, y, z = self.aruco_info.position
        self.setText(f"ArUco {self.marker_id}: {self.aruco_info.dictionary} ID:{self.aruco_info.marker_id} ({x:.3f}, {y:.3f}, {z:.3f})")
        
    def get_data(self) -> Dict[str, Any]:
        """Get marker data as dictionary."""
        data = self.aruco_info.to_dict()
        data["id"] = self.marker_id  # Add our internal marker ID
        return data


class MarkerPanel(QGroupBox):
    """Panel for managing ArUco markers."""
    
    # Signals
    marker_added = pyqtSignal(int, ArUcoMarkerInfo)   # marker_id, aruco_info
    marker_removed = pyqtSignal(int)                  # marker_id
    marker_selected = pyqtSignal(int)                 # marker_id
    marker_position_changed = pyqtSignal(int, tuple)  # marker_id, new_position
    marker_moved = pyqtSignal(int, tuple)             # marker_id, new_position
    placement_mode_requested = pyqtSignal(bool)       # enable/disable placement mode
    
    def __init__(self):
        super().__init__("ArUco Markers")
        self.next_marker_id = 0
        self.markers: Dict[int, MarkerListItem] = {}
        self.aruco_generator = ArUcoGenerator()
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # ArUco Configuration section
        aruco_config_group = QGroupBox("ArUco Configuration")
        aruco_layout = QVBoxLayout(aruco_config_group)
        
        # Dictionary selection
        dict_layout = QHBoxLayout()
        dict_layout.addWidget(QLabel("Dictionary:"))
        self.dict_combo = QComboBox()
        self.dict_combo.addItems(ArUcoGenerator.get_available_dictionaries())
        self.dict_combo.setCurrentText("DICT_4X4_50")  # Default to 4x4_50
        self.dict_combo.setToolTip("ArUco dictionary to use")
        dict_layout.addWidget(self.dict_combo)
        aruco_layout.addLayout(dict_layout)
        
        # Marker ID selection
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Marker ID:"))
        self.marker_id_spinbox = QSpinBox()
        self.marker_id_spinbox.setRange(0, 49)  # Default for 4x4_50
        self.marker_id_spinbox.setValue(0)
        self.marker_id_spinbox.setToolTip("ID of the ArUco marker (must be unique)")
        id_layout.addWidget(self.marker_id_spinbox)
        aruco_layout.addLayout(id_layout)
        
        # Marker size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.01, 1.0)
        self.size_spinbox.setValue(0.05)
        self.size_spinbox.setSingleStep(0.01)
        self.size_spinbox.setDecimals(3)
        self.size_spinbox.setSuffix(" m")
        self.size_spinbox.setToolTip("Physical size of the ArUco marker")
        size_layout.addWidget(self.size_spinbox)
        aruco_layout.addLayout(size_layout)
        
        layout.addWidget(aruco_config_group)
        
        # Add marker section
        add_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add ArUco Marker")
        self.add_button.setToolTip("Click to place a new ArUco marker on the 3D model")
        add_layout.addWidget(self.add_button)
        
        layout.addLayout(add_layout)
        
        # Marker list
        list_label = QLabel("Placed Markers:")
        layout.addWidget(list_label)
        
        self.marker_list = QListWidget()
        self.marker_list.setMaximumHeight(150)
        self.marker_list.setToolTip("List of placed ArUco markers")
        layout.addWidget(self.marker_list)
        
        # Marker controls
        controls_layout = QHBoxLayout()
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.setEnabled(False)
        self.edit_button.setToolTip("Edit selected marker properties")
        controls_layout.addWidget(self.edit_button)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip("Remove selected marker")
        controls_layout.addWidget(self.remove_button)
        
        layout.addLayout(controls_layout)
        
        # Marker movement controls
        movement_group = QGroupBox("Marker Movement")
        movement_layout = QVBoxLayout(movement_group)
        
        # Step size control
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.step_spinbox = QDoubleSpinBox()
        self.step_spinbox.setRange(0.001, 0.1)
        self.step_spinbox.setValue(0.01)
        self.step_spinbox.setSingleStep(0.001)
        self.step_spinbox.setDecimals(3)
        self.step_spinbox.setSuffix(" m")
        self.step_spinbox.setToolTip("Distance to move marker per step")
        step_layout.addWidget(self.step_spinbox)
        movement_layout.addLayout(step_layout)
        
        # Directional buttons
        direction_layout = QVBoxLayout()
        
        # Up button
        up_layout = QHBoxLayout()
        up_layout.addStretch()
        self.up_button = QPushButton("↑")
        self.up_button.setFixedSize(40, 30)
        self.up_button.setEnabled(False)
        self.up_button.setToolTip("Move marker up")
        up_layout.addWidget(self.up_button)
        up_layout.addStretch()
        direction_layout.addLayout(up_layout)
        
        # Left, Down, Right buttons
        ldr_layout = QHBoxLayout()
        self.left_button = QPushButton("←")
        self.left_button.setFixedSize(40, 30)
        self.left_button.setEnabled(False)
        self.left_button.setToolTip("Move marker left")
        ldr_layout.addWidget(self.left_button)
        
        self.down_button = QPushButton("↓")
        self.down_button.setFixedSize(40, 30)
        self.down_button.setEnabled(False)
        self.down_button.setToolTip("Move marker down")
        ldr_layout.addWidget(self.down_button)
        
        self.right_button = QPushButton("→")
        self.right_button.setFixedSize(40, 30)
        self.right_button.setEnabled(False)
        self.right_button.setToolTip("Move marker right")
        ldr_layout.addWidget(self.right_button)
        direction_layout.addLayout(ldr_layout)
        
        movement_layout.addLayout(direction_layout)
        layout.addWidget(movement_group)
        
        # Position input section (for manual entry)
        position_group = QGroupBox("Manual Position Entry")
        position_layout = QVBoxLayout(position_group)
        
        coords_layout = QHBoxLayout()
        
        # X coordinate
        coords_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-10.0, 10.0)
        self.x_spinbox.setDecimals(3)
        self.x_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.x_spinbox)
        
        # Y coordinate
        coords_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-10.0, 10.0)
        self.y_spinbox.setDecimals(3)
        self.y_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.y_spinbox)
        
        # Z coordinate
        coords_layout.addWidget(QLabel("Z:"))
        self.z_spinbox = QDoubleSpinBox()
        self.z_spinbox.setRange(-10.0, 10.0)
        self.z_spinbox.setDecimals(3)
        self.z_spinbox.setSuffix(" m")
        coords_layout.addWidget(self.z_spinbox)
        
        position_layout.addLayout(coords_layout)
        
        self.place_at_coords_button = QPushButton("Place at Coordinates")
        self.place_at_coords_button.setToolTip("Place marker at manually entered coordinates")
        position_layout.addWidget(self.place_at_coords_button)
        
        layout.addWidget(position_group)
        
        # Statistics
        self.stats_label = QLabel("Markers: 0")
        self.stats_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.stats_label)
        
    def setup_connections(self) -> None:
        """Setup signal-slot connections."""
        self.add_button.clicked.connect(self.enter_placement_mode)
        self.place_at_coords_button.clicked.connect(self.place_marker_at_coordinates)
        self.edit_button.clicked.connect(self.edit_selected_marker)
        self.remove_button.clicked.connect(self.remove_selected_marker)
        
        # Connect movement buttons
        self.up_button.clicked.connect(lambda: self.move_marker('up'))
        self.down_button.clicked.connect(lambda: self.move_marker('down'))
        self.left_button.clicked.connect(lambda: self.move_marker('left'))
        self.right_button.clicked.connect(lambda: self.move_marker('right'))
        
        self.marker_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.marker_list.itemDoubleClicked.connect(self.edit_selected_marker)
        
        # Connect spinbox changes to update marker position
        self.x_spinbox.valueChanged.connect(self.on_coordinate_changed)
        self.y_spinbox.valueChanged.connect(self.on_coordinate_changed)
        self.z_spinbox.valueChanged.connect(self.on_coordinate_changed)
        
        # Connect ArUco configuration changes
        self.dict_combo.currentTextChanged.connect(self.on_dictionary_changed)
        
    def on_dictionary_changed(self) -> None:
        """Update marker ID range when dictionary changes."""
        current_dict = self.dict_combo.currentText()
        max_id = ArUcoGenerator.get_max_id_for_dict(current_dict)
        self.marker_id_spinbox.setRange(0, max_id)
        self.marker_id_spinbox.setToolTip(f"ID of the ArUco marker (0-{max_id} for {current_dict})")
        
        # Reset to 0 if current value is out of range
        if self.marker_id_spinbox.value() > max_id:
            self.marker_id_spinbox.setValue(0)
        
    def enter_placement_mode(self) -> None:
        """Enter marker placement mode."""
        # Emit signal to tell the viewer to enter placement mode
        self.placement_mode_requested.emit(True)
        
        # Update button text to indicate placement mode
        self.add_button.setText("Cancel Placement")
        self.add_button.clicked.disconnect()
        self.add_button.clicked.connect(self.cancel_placement_mode)
        
        # Disable other controls during placement
        self.place_at_coords_button.setEnabled(False)
        self.edit_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        
    def cancel_placement_mode(self) -> None:
        """Cancel marker placement mode."""
        # Emit signal to tell the viewer to exit placement mode
        self.placement_mode_requested.emit(False)
        
        # Restore button text and connection
        self.add_button.setText("Add Marker")
        self.add_button.clicked.disconnect()
        self.add_button.clicked.connect(self.enter_placement_mode)
        
        # Re-enable controls
        self.place_at_coords_button.setEnabled(True)
        current_item = self.marker_list.currentItem()
        has_selection = current_item is not None
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        
    def place_marker_at_coordinates(self) -> None:
        """Place marker at manually entered coordinates."""
        position = (
            self.x_spinbox.value(),
            self.y_spinbox.value(),
            self.z_spinbox.value()
        )
        self.place_marker_at_position(position)
        
    def place_marker_at_position(self, position: tuple) -> None:
        """Place a marker at the specified position."""
        marker_id = self.next_marker_id
        
        # Get ArUco configuration
        dictionary = self.dict_combo.currentText()
        aruco_id = self.marker_id_spinbox.value()
        size = self.size_spinbox.value()
        
        # Check if this ArUco ID is already used
        for existing_item in self.markers.values():
            if (existing_item.aruco_info.dictionary == dictionary and 
                existing_item.aruco_info.marker_id == aruco_id):
                QMessageBox.warning(
                    self,
                    "Duplicate ArUco Marker",
                    f"ArUco marker {dictionary} ID:{aruco_id} is already placed.\n"
                    f"Each ArUco marker must have a unique dictionary-ID combination."
                )
                return
        
        # Note: The marker position becomes the origin (0,0,0) for the object coordinate system
        # This is important for robotics applications where grasp poses are relative to the marker
        
        # Create ArUco marker info
        aruco_info = ArUcoMarkerInfo(
            dictionary=dictionary,
            marker_id=aruco_id,
            position=position,
            size=size
        )
        
        # Create list item
        item = MarkerListItem(marker_id, aruco_info)
        self.markers[marker_id] = item
        
        # Add to list
        self.marker_list.addItem(item)
        
        # Emit signal
        self.marker_added.emit(marker_id, aruco_info)
        
        # Update stats
        self.update_stats()
        
        # Increment ID for next marker
        self.next_marker_id += 1
        
        # Auto-increment ArUco ID for convenience (if not at max)
        max_id = ArUcoGenerator.get_max_id_for_dict(dictionary)
        if aruco_id < max_id:
            self.marker_id_spinbox.setValue(aruco_id + 1)
        
    def place_marker_at_clicked_position(self, position: tuple) -> None:
        """Place a marker at a position clicked in the 3D viewer."""
        self.place_marker_at_position(position)
        # Exit placement mode after placing
        self.cancel_placement_mode()
        
    def remove_selected_marker(self) -> None:
        """Remove the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            reply = QMessageBox.question(
                self,
                "Remove Marker",
                f"Remove Marker {current_item.marker_id}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Remove from list
                row = self.marker_list.row(current_item)
                self.marker_list.takeItem(row)
                
                # Remove from dictionary
                marker_id = current_item.marker_id
                del self.markers[marker_id]
                
                # Emit signal
                self.marker_removed.emit(marker_id)
                
                # Update stats
                self.update_stats()
                
    def edit_selected_marker(self) -> None:
        """Edit the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            # Get new position
            x, y, z = current_item.aruco_info.position
            
            new_x, ok1 = QInputDialog.getDouble(
                self, "Edit X Position", "X coordinate:", x, -10.0, 10.0, 3
            )
            if not ok1:
                return
                
            new_y, ok2 = QInputDialog.getDouble(
                self, "Edit Y Position", "Y coordinate:", y, -10.0, 10.0, 3
            )
            if not ok2:
                return
                
            new_z, ok3 = QInputDialog.getDouble(
                self, "Edit Z Position", "Z coordinate:", z, -10.0, 10.0, 3
            )
            if not ok3:
                return
                
            # Update position
            new_position = (new_x, new_y, new_z)
            current_item.aruco_info.position = new_position
            current_item.update_text()
            
            # Emit signal
            self.marker_position_changed.emit(current_item.marker_id, new_position)
            
            # Update the coordinate spinboxes to reflect the change
            self.x_spinbox.setValue(new_x)
            self.y_spinbox.setValue(new_y)
            self.z_spinbox.setValue(new_z)
            
    def on_selection_changed(self) -> None:
        """Handle marker selection changes."""
        current_item = self.marker_list.currentItem()
        has_selection = current_item is not None
        
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        self.up_button.setEnabled(has_selection)
        self.down_button.setEnabled(has_selection)
        self.left_button.setEnabled(has_selection)
        self.right_button.setEnabled(has_selection)
        
        if has_selection and isinstance(current_item, MarkerListItem):
            # Emit selection signal
            self.marker_selected.emit(current_item.marker_id)
            
            # Update coordinate spinboxes
            x, y, z = current_item.aruco_info.position
            self.x_spinbox.setValue(x)
            self.y_spinbox.setValue(y)
            self.z_spinbox.setValue(z)
    
    def move_marker(self, direction: str) -> None:
        """Move the selected marker in the specified direction."""
        current_item = self.marker_list.currentItem()
        if not current_item or not isinstance(current_item, MarkerListItem):
            return
        
        # Get current position and step size
        x, y, z = current_item.aruco_info.position
        step = self.step_spinbox.value()
        
        # Calculate new position based on direction
        if direction == 'up':
            new_position = (x, y + step, z)
        elif direction == 'down':
            new_position = (x, y - step, z)
        elif direction == 'left':
            new_position = (x - step, y, z)
        elif direction == 'right':
            new_position = (x + step, y, z)
        else:
            return
        
        # Update the marker's position
        current_item.aruco_info.position = new_position
        current_item.update_text()
        
        # Update coordinate spinboxes
        self.x_spinbox.setValue(new_position[0])
        self.y_spinbox.setValue(new_position[1])
        self.z_spinbox.setValue(new_position[2])
        
        # Emit movement signal
        self.marker_moved.emit(current_item.marker_id, new_position)
            
    def on_coordinate_changed(self) -> None:
        """Handle coordinate spinbox changes for real-time marker movement."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            # Get new position from spinboxes
            new_position = (
                self.x_spinbox.value(),
                self.y_spinbox.value(),
                self.z_spinbox.value()
            )
            
            # Update the marker item
            current_item.aruco_info.position = new_position
            current_item.update_text()
            
            # Emit signal to update 3D viewer
            self.marker_position_changed.emit(current_item.marker_id, new_position)
            
    def update_stats(self) -> None:
        """Update statistics display."""
        count = len(self.markers)
        self.stats_label.setText(f"Markers: {count}")
        
    def get_all_markers(self) -> List[Dict[str, Any]]:
        """Get all markers as a list of dictionaries."""
        return [item.get_data() for item in self.markers.values()]
        
    def load_markers(self, markers_data: List[Dict[str, Any]]) -> None:
        """Load markers from data."""
        # Clear existing markers
        self.marker_list.clear()
        self.markers.clear()
        
        # Load each marker
        for marker_data in markers_data:
            marker_id = marker_data["id"]
            
            # Create ArUcoMarkerInfo from data
            if "dictionary" in marker_data:
                # New format with ArUco info
                aruco_info = ArUcoMarkerInfo.from_dict(marker_data)
            else:
                # Legacy format - convert to ArUco info
                position = tuple(marker_data["position"])
                size = marker_data.get("size", 0.05)
                aruco_info = ArUcoMarkerInfo(
                    dictionary="DICT_4X4_50",  # Default
                    marker_id=0,  # Default
                    position=position,
                    size=size
                )
            
            # Create item
            item = MarkerListItem(marker_id, aruco_info)
            self.markers[marker_id] = item
            self.marker_list.addItem(item)
            
            # Update next ID
            self.next_marker_id = max(self.next_marker_id, marker_id + 1)
            
        # Update stats
        self.update_stats()
        
    def clear_all_markers(self) -> None:
        """Clear all markers."""
        reply = QMessageBox.question(
            self,
            "Clear All Markers",
            "Remove all markers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove all markers
            for marker_id in list(self.markers.keys()):
                self.marker_removed.emit(marker_id)
                
            self.marker_list.clear()
            self.markers.clear()
            self.update_stats()
            
    def get_selected_marker_id(self) -> int:
        """Get the ID of the currently selected marker."""
        current_item = self.marker_list.currentItem()
        if current_item and isinstance(current_item, MarkerListItem):
            return current_item.marker_id
        return -1
