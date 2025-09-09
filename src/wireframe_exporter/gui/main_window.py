#!/usr/bin/env python3
"""
Main window for the Wireframe Exporter GUI application.
"""

import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QTextEdit, QComboBox, QGroupBox, QGridLayout, QLineEdit,
    QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

import open3d as o3d
import numpy as np

# Add the scripts directory to the path
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from export_wireframe import WireframeExporter
from plot_wireframe_example import plot_wireframe_json


class WireframeExportThread(QThread):
    """Thread for wireframe export operations."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, bool)  # message, success
    error = pyqtSignal(str)
    
    def __init__(self, input_file: str, output_format: str, output_path: str, auto_export: bool = False):
        super().__init__()
        self.input_file = input_file
        self.output_format = output_format
        self.output_path = output_path
        self.auto_export = auto_export
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Create exporter
            exporter = WireframeExporter()
            
            self.progress.emit(30)
            
            # Load mesh
            exporter.load_mesh(self.input_file)
            
            self.progress.emit(50)
            
            # Export wireframe based on format
            if self.output_format == 'json':
                success = exporter.export_json(self.output_path)
            elif self.output_format == 'csv':
                success = exporter.export_csv(self.output_path)
            elif self.output_format == 'numpy':
                success = exporter.export_numpy(self.output_path)
            elif self.output_format == 'ply':
                success = exporter.export_ply(self.output_path)
            elif self.output_format == 'obj':
                success = exporter.export_obj(self.output_path)
            else:
                raise ValueError(f"Unsupported format: {self.output_format}")
            
            self.progress.emit(80)
            
            if success:
                if self.auto_export:
                    # Also export to examples folder
                    examples_dir = Path(__file__).parent.parent / "examples"
                    examples_dir.mkdir(exist_ok=True)
                    examples_path = examples_dir / f"{Path(self.input_file).stem}_wireframe.{self.output_format}"
                    
                    # Export to examples folder using the same format
                    if self.output_format == 'json':
                        exporter.export_json(str(examples_path))
                    elif self.output_format == 'csv':
                        exporter.export_csv(str(examples_path))
                    elif self.output_format == 'numpy':
                        exporter.export_numpy(str(examples_path))
                    elif self.output_format == 'ply':
                        exporter.export_ply(str(examples_path))
                    elif self.output_format == 'obj':
                        exporter.export_obj(str(examples_path))
                
                self.progress.emit(100)
                self.finished.emit(f"Wireframe exported successfully to {self.output_path}", True)
            else:
                self.error.emit("Failed to export wireframe")
                
        except Exception as e:
            self.error.emit(f"Export error: {str(e)}")


class WireframeViewerWidget(QWidget):
    """Widget for displaying wireframe visualization."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_wireframe_file = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_panel = QGroupBox("Wireframe Viewer Controls")
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Wireframe File")
        self.load_btn.clicked.connect(self.load_wireframe_file)
        
        self.view_btn = QPushButton("View Wireframe")
        self.view_btn.clicked.connect(self.view_wireframe)
        self.view_btn.setEnabled(False)
        
        self.export_viewer_btn = QPushButton("Export Current View")
        self.export_viewer_btn.clicked.connect(self.export_current_view)
        self.export_viewer_btn.setEnabled(False)
        
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.view_btn)
        control_layout.addWidget(self.export_viewer_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)
        
        # Status display
        self.status_label = QLabel("No wireframe file loaded")
        layout.addWidget(self.status_label)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        self.setLayout(layout)
    
    def load_wireframe_file(self):
        """Load a wireframe file for viewing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Wireframe File",
            str(Path.home()),
            "Wireframe Files (*.json *.csv *.npy);;JSON Files (*.json);;CSV Files (*.csv);;NumPy Files (*.npy)"
        )
        
        if file_path:
            self.current_wireframe_file = file_path
            self.view_btn.setEnabled(True)
            self.export_viewer_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            
            # Display file info
            self.display_file_info(file_path)
    
    def display_file_info(self, file_path: str):
        """Display information about the loaded wireframe file."""
        try:
            if file_path.endswith('.json'):
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                info = f"""File: {Path(file_path).name}
Format: JSON
Vertices: {data.get('mesh_info', {}).get('num_vertices', 'Unknown')}
Edges: {data.get('mesh_info', {}).get('num_edges', 'Unknown')}
Bounding Box: {data.get('mesh_info', {}).get('bounding_box', {})}
"""
            else:
                info = f"File: {Path(file_path).name}\nFormat: {Path(file_path).suffix.upper()}\n(Detailed info not available for this format)"
            
            self.info_text.setText(info)
            
        except Exception as e:
            self.info_text.setText(f"Error reading file info: {str(e)}")
    
    def view_wireframe(self):
        """Open wireframe visualization."""
        if self.current_wireframe_file:
            try:
                if self.current_wireframe_file.endswith('.json'):
                    plot_wireframe_json(self.current_wireframe_file)
                else:
                    QMessageBox.warning(self, "Warning", "Only JSON wireframe files are supported for viewing in this version.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to view wireframe: {str(e)}")
    
    def export_current_view(self):
        """Export current wireframe view to examples folder."""
        if self.current_wireframe_file:
            try:
                examples_dir = Path(__file__).parent.parent / "examples"
                examples_dir.mkdir(exist_ok=True)
                
                # Copy current file to examples with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{Path(self.current_wireframe_file).stem}_view_{timestamp}{Path(self.current_wireframe_file).suffix}"
                new_path = examples_dir / new_name
                
                import shutil
                shutil.copy2(self.current_wireframe_file, new_path)
                
                QMessageBox.information(self, "Success", f"Wireframe view exported to:\n{new_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export view: {str(e)}")


class WireframeExporterMainWindow(QMainWindow):
    """Main window for the Wireframe Exporter GUI application."""
    
    def __init__(self):
        super().__init__()
        self.current_mesh_file = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Wireframe Exporter - 3D Mesh Wireframe Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Export tab
        self.export_tab = self.create_export_tab()
        self.tab_widget.addTab(self.export_tab, "Export Wireframe")
        
        # Viewer tab
        self.viewer_tab = WireframeViewerWidget()
        self.tab_widget.addTab(self.viewer_tab, "View Wireframe")
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_export_tab(self):
        """Create the export tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Model loading section
        model_group = QGroupBox("3D Model Loading")
        model_layout = QGridLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select a 3D model file...")
        self.model_path_edit.setReadOnly(True)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_model_file)
        
        self.model_info_label = QLabel("No model loaded")
        
        model_layout.addWidget(QLabel("Model File:"), 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(self.browse_btn, 0, 2)
        model_layout.addWidget(self.model_info_label, 1, 0, 1, 3)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Export options section
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout()
        
        # Format selection
        export_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["json", "csv", "numpy", "ply", "obj"])
        self.format_combo.setCurrentText("json")
        export_layout.addWidget(self.format_combo, 0, 1)
        
        # Output path
        export_layout.addWidget(QLabel("Output Path:"), 1, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Auto-generated if empty")
        export_layout.addWidget(self.output_path_edit, 1, 1)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_file)
        export_layout.addWidget(self.output_browse_btn, 1, 2)
        
        # Auto-export to examples
        self.auto_export_check = QCheckBox("Auto-export to examples folder")
        self.auto_export_check.setChecked(True)
        export_layout.addWidget(self.auto_export_check, 2, 0, 1, 3)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Export controls
        control_group = QGroupBox("Export Controls")
        control_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Wireframe")
        self.export_btn.clicked.connect(self.export_wireframe)
        self.export_btn.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Log display
        log_group = QGroupBox("Export Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def browse_model_file(self):
        """Browse for a 3D model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D Model File",
            str(Path.home()),
            "3D Model Files (*.obj *.stl *.ply *.off);;OBJ Files (*.obj);;STL Files (*.stl);;PLY Files (*.ply);;OFF Files (*.off)"
        )
        
        if file_path:
            self.current_mesh_file = file_path
            self.model_path_edit.setText(file_path)
            self.export_btn.setEnabled(True)
            
            # Display model info
            self.display_model_info(file_path)
            
            # Auto-generate output path
            if not self.output_path_edit.text():
                output_path = Path(file_path).with_suffix(f".wireframe.{self.format_combo.currentText()}")
                self.output_path_edit.setText(str(output_path))
    
    def browse_output_file(self):
        """Browse for output file location."""
        if not self.current_mesh_file:
            QMessageBox.warning(self, "Warning", "Please select a model file first.")
            return
        
        format_ext = self.format_combo.currentText()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Wireframe As",
            str(Path.home()),
            f"{format_ext.upper()} Files (*.{format_ext})"
        )
        
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def display_model_info(self, file_path: str):
        """Display information about the loaded model."""
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            if len(mesh.vertices) == 0:
                self.model_info_label.setText("Error: Could not load mesh")
                self.export_btn.setEnabled(False)
                return
            
            bbox = mesh.get_axis_aligned_bounding_box()
            size = bbox.max_bound - bbox.min_bound
            
            info = f"""Vertices: {len(mesh.vertices)} | Triangles: {len(mesh.triangles)} | 
            Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units"""
            
            self.model_info_label.setText(info)
            self.log_text.append(f"Loaded model: {Path(file_path).name}")
            self.log_text.append(f"  - Vertices: {len(mesh.vertices)}")
            self.log_text.append(f"  - Triangles: {len(mesh.triangles)}")
            self.log_text.append(f"  - Bounding box: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
            
        except Exception as e:
            self.model_info_label.setText(f"Error loading model: {str(e)}")
            self.export_btn.setEnabled(False)
            self.log_text.append(f"Error loading model: {str(e)}")
    
    def export_wireframe(self):
        """Export wireframe data."""
        if not self.current_mesh_file:
            QMessageBox.warning(self, "Warning", "Please select a model file first.")
            return
        
        output_path = self.output_path_edit.text().strip()
        if not output_path:
            output_path = str(Path(self.current_mesh_file).with_suffix(f".wireframe.{self.format_combo.currentText()}"))
            self.output_path_edit.setText(output_path)
        
        # Start export in thread
        self.export_thread = WireframeExportThread(
            self.current_mesh_file,
            self.format_combo.currentText(),
            output_path,
            self.auto_export_check.isChecked()
        )
        
        self.export_thread.progress.connect(self.progress_bar.setValue)
        self.export_thread.finished.connect(self.export_finished)
        self.export_thread.error.connect(self.export_error)
        
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_text.append(f"Starting export to {output_path}...")
        self.export_thread.start()
    
    def export_finished(self, message: str, success: bool):
        """Handle export completion."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(message)
        self.statusBar().showMessage(message)
        
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Switch to viewer tab and load the exported file
            self.tab_widget.setCurrentIndex(1)  # Switch to viewer tab
            self.viewer_tab.current_wireframe_file = self.output_path_edit.text()
            self.viewer_tab.view_btn.setEnabled(True)
            self.viewer_tab.export_viewer_btn.setEnabled(True)
            self.viewer_tab.status_label.setText(f"Loaded: {Path(self.output_path_edit.text()).name}")
            self.viewer_tab.display_file_info(self.output_path_edit.text())
    
    def export_error(self, error_message: str):
        """Handle export error."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log_text.append(f"ERROR: {error_message}")
        self.statusBar().showMessage(f"Export failed: {error_message}")
        
        QMessageBox.critical(self, "Export Error", error_message)


def main():
    """Main function to run the GUI application."""
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setApplicationName("Wireframe Exporter")
    app.setApplicationVersion("1.0.0")
    
    window = WireframeExporterMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
