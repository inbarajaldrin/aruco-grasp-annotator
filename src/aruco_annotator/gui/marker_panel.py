"""Panel for managing ArUco markers on the 3D model."""

from typing import List, Dict, Any
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel,
    QSpinBox, QDoubleSpinBox, QInputDialog, QMessageBox
)


class MarkerListItem(QListWidgetItem):
    """Custom list item for ArUco markers."""
    
    def __init__(self, marker_id: int, position: tuple, size: float = 0.05):
        super().__init__()
        self.marker_id = marker_id
        self.position = position
        self.size = size
        self.update_text()
        
    def update_text(self) -> None:
        """Update the display text."""
        x, y, z = self.position
        self.setText(f"Marker {self.marker_id}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
    def get_data(self) -> Dict[str, Any]:
        """Get marker data as dictionary."""
        return {
            "id": self.marker_id,
            "position": list(self.position),
            "size": self.size
        }


class MarkerPanel(QGroupBox):
    """Panel for managing ArUco markers."""
    
    # Signals
    marker_added = pyqtSignal(int, tuple, float)      # marker_id, position, size
    marker_removed = pyqtSignal(int)                  # marker_id
    marker_selected = pyqtSignal(int)                 # marker_id
    marker_position_changed = pyqtSignal(int, tuple)  # marker_id, new_position
    
    def __init__(self):
        super().__init__("ArUco Markers")
        self.next_marker_id = 0
        self.markers: Dict[int, MarkerListItem] = {}
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Add marker section
        add_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Marker")
        self.add_button.setToolTip("Click to place a new ArUco marker")
        add_layout.addWidget(self.add_button)
        
        # Marker size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setRange(0.01, 1.0)
        self.size_spinbox.setValue(0.05)
        self.size_spinbox.setSingleStep(0.01)
        self.size_spinbox.setDecimals(3)
        self.size_spinbox.setSuffix(" m")
        self.size_spinbox.setToolTip("Size of the ArUco marker")
        size_layout.addWidget(self.size_spinbox)
        
        layout.addLayout(add_layout)
        layout.addLayout(size_layout)
        
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
        
        self.marker_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.marker_list.itemDoubleClicked.connect(self.edit_selected_marker)
        
    def enter_placement_mode(self) -> None:
        """Enter marker placement mode."""
        # This would typically change the 3D viewer to placement mode
        # For now, we'll just place at origin as an example
        self.place_marker_at_position((0.0, 0.0, 0.0))
        
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
        size = self.size_spinbox.value()
        
        # Create list item
        item = MarkerListItem(marker_id, position, size)
        self.markers[marker_id] = item
        
        # Add to list
        self.marker_list.addItem(item)
        
        # Emit signal
        self.marker_added.emit(marker_id, position, size)
        
        # Update stats
        self.update_stats()
        
        # Increment ID for next marker
        self.next_marker_id += 1
        
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
            x, y, z = current_item.position
            
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
            current_item.position = new_position
            current_item.update_text()
            
            # Emit signal
            self.marker_position_changed.emit(current_item.marker_id, new_position)
            
    def on_selection_changed(self) -> None:
        """Handle marker selection changes."""
        current_item = self.marker_list.currentItem()
        has_selection = current_item is not None
        
        self.edit_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        
        if has_selection and isinstance(current_item, MarkerListItem):
            # Emit selection signal
            self.marker_selected.emit(current_item.marker_id)
            
            # Update coordinate spinboxes
            x, y, z = current_item.position
            self.x_spinbox.setValue(x)
            self.y_spinbox.setValue(y)
            self.z_spinbox.setValue(z)
            
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
            position = tuple(marker_data["position"])
            size = marker_data.get("size", 0.05)
            
            # Create item
            item = MarkerListItem(marker_id, position, size)
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
