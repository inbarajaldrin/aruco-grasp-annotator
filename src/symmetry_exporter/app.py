#!/usr/bin/env python3
"""
Symmetry Exporter - 3D Visualization App
Interactive 3D environment for displaying individual objects with grasp points
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np

app = FastAPI(
    title="Symmetry Exporter",
    description="Interactive 3D visualization tool for objects with grasp points",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Available components
COMPONENTS = [
    "base_scaled70",
    "fork_orange_scaled70", 
    "fork_yellow_scaled70",
    "line_brown_scaled70",
    "line_red_scaled70"
]

# Global state
app_state = {
    "components": {},
    "next_id": 1
}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the complete 3D visualization interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Symmetry Exporter - 3D Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                height: 100vh;
                overflow: hidden;
            }
            
            .app-container {
                display: flex;
                height: 100vh;
            }
            
            .sidebar {
                width: 400px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 20px;
                overflow-y: auto;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                border-right: 1px solid #ddd;
                z-index: 10;
                position: relative;
            }
            
            .main-viewer {
                flex: 1;
                position: relative;
                display: flex;
                flex-direction: column;
            }
            
            .viewer-container {
                flex: 1;
                background: #1a1a1a;
                position: relative;
            }
            
            .controls-panel {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }
            
            .controls-panel h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 16px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .component-list {
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            
            .component-item {
                padding: 12px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .component-item:hover {
                background: #f8f9fa;
                transform: translateX(2px);
            }
            
            .component-item:last-child {
                border-bottom: none;
            }
            
            .component-name {
                font-weight: 500;
                color: #2c3e50;
            }
            
            .component-status {
                font-size: 12px;
                padding: 3px 8px;
                border-radius: 12px;
                background: #e0e0e0;
                color: #666;
            }
            
            .component-status.loaded {
                background: #27ae60;
                color: white;
            }
            
            .btn {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
                margin: 3px;
                min-width: 100px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .btn-secondary {
                background: linear-gradient(45deg, #e74c3c, #c0392b);
            }
            
            .btn-secondary:hover {
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            }
            
            .btn-small {
                padding: 6px 12px;
                font-size: 12px;
                min-width: 60px;
            }
            
            .control-buttons-group {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .control-buttons-group .btn {
                width: 100%;
                margin: 0;
            }
            
            .precision-controls {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                border: 2px solid #3498db;
            }
            
            .control-group {
                margin-bottom: 20px;
            }
            
            .control-group h4 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .control-row {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                gap: 8px;
                min-width: 0;
            }
            
            .control-label {
                min-width: 20px;
                font-weight: 500;
                color: #34495e;
                font-size: 13px;
            }
            
            .control-input {
                flex: 1;
                padding: 6px 10px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 13px;
                background: white;
            }
            
            .control-input:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            
            .control-buttons {
                display: flex;
                gap: 4px;
                flex-shrink: 0;
            }
            
            .selected-object {
                background: rgba(52, 152, 219, 0.1);
                border: 2px solid #3498db;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
            }
            
            .selected-object h4 {
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .selected-object p {
                color: #7f8c8d;
                font-size: 12px;
                margin: 0;
            }
            
            .status-bar {
                background: rgba(44, 62, 80, 0.95);
                color: white;
                padding: 12px 20px;
                font-size: 14px;
                backdrop-filter: blur(10px);
                border-top: 1px solid #34495e;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .status-info {
                display: flex;
                gap: 20px;
                font-size: 13px;
            }
            
            .instructions {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                font-size: 13px;
                line-height: 1.4;
                color: #555;
                border-left: 4px solid #3498db;
            }
            
            .instructions h4 {
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 14px;
            }
            
            .instructions ul {
                margin-left: 15px;
                margin-top: 5px;
            }
            
            .instructions li {
                margin-bottom: 3px;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .error {
                background: #fdf2f2;
                color: #e53e3e;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #e53e3e;
                font-size: 13px;
            }
            
            .success {
                background: #f0fff4;
                color: #38a169;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #38a169;
                font-size: 13px;
            }
            
            .info {
                background: #ebf8ff;
                color: #2b6cb0;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                border-left: 4px solid #2b6cb0;
                font-size: 13px;
            }
            
            .scene-objects {
                max-height: 150px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            
            .scene-object-item {
                padding: 8px 12px;
                border-bottom: 1px solid #eee;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 13px;
            }
            
            .scene-object-item:hover {
                background: #f8f9fa;
            }
            
            .scene-object-item.selected {
                background: rgba(52, 152, 219, 0.1);
                border-left: 3px solid #3498db;
            }
            
            .scene-object-item:last-child {
                border-bottom: none;
            }
            
            .object-name {
                font-weight: 500;
                color: #2c3e50;
            }
            
            .object-type {
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 8px;
                background: #ecf0f1;
                color: #7f8c8d;
            }
            
            .quick-actions {
                display: flex;
                gap: 5px;
                margin-top: 10px;
            }
            
            .axis-controls {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 5px;
                margin-top: 8px;
            }
            
            .axis-btn {
                padding: 4px 8px;
                border: 1px solid #bdc3c7;
                background: white;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .axis-btn:hover {
                background: #ecf0f1;
            }
            
            .axis-btn.x { border-color: #e74c3c; color: #e74c3c; }
            .axis-btn.y { border-color: #27ae60; color: #27ae60; }
            .axis-btn.z { border-color: #3498db; color: #3498db; }
            
            .floating-controls {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                backdrop-filter: blur(10px);
                display: none;
                z-index: 1000;
            }
            
            .floating-controls h4 {
                margin: 0 0 10px 0;
                color: #2c3e50;
            }
            
            .floating-controls .controls-grid {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .show-controls-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(52, 152, 219, 0.9);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 16px;
                width: 40px;
                height: 40px;
                z-index: 999;
                transition: all 0.2s;
            }
            
            .show-controls-btn:hover {
                background: rgba(52, 152, 219, 1);
                transform: scale(1.1);
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="instructions">
                    <h4>Symmetry Exporter Instructions</h4>
                    <ul>
                        <li><strong>Load:</strong> Click "Load All Components" to load parts</li>
                        <li><strong>Add:</strong> Click component names to add to scene</li>
                        <li><strong>Select:</strong> Click objects in scene to select</li>
                        <li><strong>Rotate:</strong> Use rotation controls (X, Y, Z) to rotate object</li>
                        <li><strong>Quaternion:</strong> View quaternion values that update as you rotate</li>
                        <li><strong>Camera:</strong> Mouse wheel zoom, middle-click pan, right-drag orbit</li>
                        <li><strong>Shortcuts:</strong> Q/E rotate Y axis, Delete removes object</li>
                    </ul>
                </div>
                
                <div class="controls-panel">
                    <h3>Components</h3>
                    <div class="control-buttons-group">
                        <button class="btn" onclick="loadAllComponents()">Load All Components</button>
                        <button class="btn btn-secondary" onclick="clearScene()">Clear Scene</button>
                    </div>
                    <div id="componentList" class="component-list">
                        <div class="loading">Click "Load All Components" to start</div>
                    </div>
                </div>
                
                <div id="selectedObjectInfo"></div>
                
                <div id="statusMessages"></div>
            </div>
            
            <div class="main-viewer">
                <div class="viewer-container" id="viewer"></div>
                <div class="status-bar" id="statusBar">
                    <span>Ready - Load components to start</span>
                    <div class="status-info">
                        <span id="objectCount">Objects: 0</span>
                        <span id="selectedInfo">Selected: None</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Floating control panel -->
        <div id="floatingControls" class="floating-controls">
            <h4>View Tools</h4>
            <div class="controls-grid">
                <button class="btn btn-small" onclick="toggleGrid()">Toggle Grid</button>
                <button class="btn btn-small" onclick="resetCamera()">Reset Camera</button>
                <button class="btn btn-small btn-secondary" onclick="hideFloatingControls()">Close</button>
            </div>
        </div>
        
        <button id="showControlsBtn" class="show-controls-btn" onclick="showFloatingControls()" title="View Tools">⚙</button>

        <script>
            // Global variables
            let scene, camera, renderer, controls;
            let loadedComponents = {};
            let sceneObjects = [];
            let selectedObject = null;
            let gridVisible = true;
            let gridHelper;
            let objectAxesHelper = null;  // Axes helper attached to selected object
            let raycaster, mouse;
            let isMouseDown = false;
            let graspPointsGroup = null;
            let graspPointsData = null;
            let currentObjectName = null;  // Track currently loaded object
            let graspPointsList = [];  // List of grasp point objects
            let currentWireframe = null;  // Reference to current wireframe object
            
            // Grasp point colors (matching grasp_candidates visualization)
            const GRASP_COLOR_DEFAULT = 0x00ff00;  // Green
            const GRASP_COLOR_SELECTED = 0xff0000;  // Red
            const GRIPPER_COLOR_DEFAULT = 0xffaa00;  // Orange
            const GRIPPER_COLOR_SELECTED = 0xff0000;  // Red for selected
            
            // Initialize the 3D scene
            function initScene() {
                const container = document.getElementById('viewer');
                
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                // Camera setup
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.001, 1000);
                camera.position.set(0.5, -0.5, 0.5);  // Move camera to look at scene from Y-negative
                camera.up.set(0, 0, 1);  // Set Z as up vector
                camera.lookAt(0, 0, 0);
                
                // Renderer setup
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);
                
                // Orbit Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.minDistance = 0.05;
                controls.maxDistance = 5;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, -10, 10);  // Light from above in Z-up system
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
                scene.add(hemisphereLight);
                
                // Grid
                gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
                gridHelper.rotateX(Math.PI / 2);  // Rotate 90 degrees to make it horizontal in XY plane
                scene.add(gridHelper);
                
                // Raycaster for object selection
                raycaster = new THREE.Raycaster();
                mouse = new THREE.Vector2();
                
                // Event listeners
                setupEventListeners();
                
                // Start animation loop
                animate();
                
                updateStatus("Scene initialized - Ready to load components");
            }
            
            function setupEventListeners() {
                const canvas = renderer.domElement;
                
                // Mouse events for object selection
                canvas.addEventListener('mousedown', onMouseDown);
                canvas.addEventListener('mouseup', onMouseUp);
                canvas.addEventListener('contextmenu', (e) => e.preventDefault());
                
                // Window resize
                window.addEventListener('resize', onWindowResize);
                
                // Keyboard shortcuts
                document.addEventListener('keydown', onKeyDown);
            }
            
            function onMouseDown(event) {
                isMouseDown = true;
            }
            
            function onMouseUp(event) {
                if (!isMouseDown) return;
                isMouseDown = false;
                
                // Only process clicks, not drags
                if (event.button === 0) { // Left click
                    mouse.x = (event.clientX / renderer.domElement.clientWidth) * 2 - 1;
                    mouse.y = -(event.clientY / renderer.domElement.clientHeight) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects(sceneObjects);
                    
                    if (intersects.length > 0) {
                        selectObject(intersects[0].object);
                    } else {
                        selectObject(null);
                    }
                }
            }
            
            function onKeyDown(event) {
                if (!selectedObject) return;
                
                const rotStep = event.shiftKey ? Math.PI / 18 : Math.PI / 36; // 10° or 5°
                
                switch(event.key) {
                    case 'q':
                    case 'Q':
                        selectedObject.rotation.y -= rotStep;
                        break;
                    case 'e':
                    case 'E':
                        selectedObject.rotation.y += rotStep;
                        break;
                    case 'Delete':
                        deleteSelectedObject();
                        break;
                }
                
                if (['q', 'e', 'Q', 'E'].includes(event.key)) {
                    event.preventDefault();
                    updateSelectedObjectInfo();  // Update to refresh quaternion display
                }
            }
            
            function onWindowResize() {
                const container = document.getElementById('viewer');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            
            // Component loading functions
            async function loadAllComponents() {
                updateStatus("Loading all components...");
                showMessage("Loading components from server...", "info");
                
                try {
                    const response = await fetch('/api/components');
                    const components = await response.json();
                    
                    loadedComponents = components;
                    updateComponentList();
                    showMessage(`Loaded ${Object.keys(components).length} components successfully!`, "success");
                    updateStatus("All components loaded - Click on component names to add to scene");
                } catch (error) {
                    showMessage(`Error loading components: ${error.message}`, "error");
                    updateStatus("Error loading components");
                }
            }
            
            function updateComponentList() {
                const list = document.getElementById('componentList');
                list.innerHTML = '';
                
                Object.keys(loadedComponents).forEach(componentName => {
                    const component = loadedComponents[componentName];
                    const item = document.createElement('div');
                    item.className = 'component-item';
                    item.onclick = () => addComponentToScene(componentName);
                    
                    item.innerHTML = `
                        <span class="component-name">${component.display_name}</span>
                        <span class="component-status loaded">Ready</span>
                    `;
                    
                    list.appendChild(item);
                });
            }
            
            function addComponentToScene(componentName) {
                const component = loadedComponents[componentName];
                if (!component) return;
                
                // Clear scene if there's already an object
                if (sceneObjects.length > 0 || currentObjectName !== null) {
                    clearScene();
                }
                
                // Create wireframe geometry
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(component.wireframe.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                // Create edges
                const edges = component.wireframe.edges;
                const lineGeometry = new THREE.BufferGeometry();
                const lineVertices = [];
                
                edges.forEach(edge => {
                    const v1 = component.wireframe.vertices[edge[0]];
                    const v2 = component.wireframe.vertices[edge[1]];
                    lineVertices.push(...v1, ...v2);
                });
                
                lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
                
                // Create material
                const material = new THREE.LineBasicMaterial({ 
                    color: getComponentColor(componentName),
                    linewidth: 2
                });
                
                // Create mesh
                const wireframe = new THREE.LineSegments(lineGeometry, material);
                wireframe.userData = { 
                    name: componentName, 
                    type: 'component',
                    displayName: component.display_name,
                    originalColor: getComponentColor(componentName),
                    id: generateId()
                };
                
                // Position at origin (scene is cleared before adding new object)
                wireframe.position.set(0, 0, 0);
                
                scene.add(wireframe);
                sceneObjects.push(wireframe);
                
                // Store reference to current wireframe
                currentWireframe = wireframe;
                
                // Add ArUco markers if available
                if (component.aruco && component.aruco.markers) {
                    component.aruco.markers.forEach((marker, index) => {
                        addArUcoMarker(marker, wireframe, index);
                    });
                }
                
                // Automatically load grasp points for this object
                currentObjectName = componentName;
                loadGraspPointsForObject(componentName);
                
                updateStatus(`Added ${component.display_name} to scene`);
                showMessage(`Added ${component.display_name} to scene`, "success");
                
                // Auto-select the new object
                selectObject(wireframe);
            }
            
            function addArUcoMarker(markerData, parentObject, index) {
                const size = markerData.size;
                const geometry = new THREE.BoxGeometry(size, size, size * 0.1);
                const material = new THREE.MeshBasicMaterial({ 
                    color: 0xff6b6b,
                    transparent: true,
                    opacity: 0.7
                });
                
                const marker = new THREE.Mesh(geometry, material);
                
                // Position relative to parent
                const pos = markerData.pose_absolute.position;
                marker.position.set(pos.x, pos.y, pos.z);
                
                // Apply rotation - convert from roll/pitch/yaw to Three.js Euler angles
                const rot = markerData.pose_absolute.rotation;
                marker.rotation.set(rot.roll, rot.pitch, rot.yaw);
                
                marker.userData = { 
                    name: `ArUco-${markerData.aruco_id}`, 
                    type: 'marker',
                    arucoId: markerData.aruco_id,
                    parentId: parentObject.userData.id,
                    displayName: `ArUco ${markerData.aruco_id}`,
                    originalColor: 0xff6b6b,
                    id: generateId()
                };
                
                // Add marker as child of parent object
                parentObject.add(marker);
                sceneObjects.push(marker);
            }
            
            function selectObject(object) {
                // Remove axes helper from previous object
                if (objectAxesHelper && objectAxesHelper.parent) {
                    objectAxesHelper.parent.remove(objectAxesHelper);
                    objectAxesHelper = null;
                }
                
                // Deselect previous object
                if (selectedObject && selectedObject.material) {
                    // Restore original color based on object type
                    if (selectedObject.userData.type === 'grasp_point') {
                        selectedObject.material.color.setHex(GRASP_COLOR_DEFAULT);
                        selectedObject.material.emissive.setHex(GRASP_COLOR_DEFAULT);
                    } else {
                        selectedObject.material.color.setHex(selectedObject.userData.originalColor);
                    }
                }
                
                selectedObject = object;
                
                if (selectedObject) {
                    // Highlight selected object
                    if (selectedObject.material) {
                        if (selectedObject.userData.type === 'grasp_point') {
                            // Red for selected grasp points (matching grasp_candidates)
                            selectedObject.material.color.setHex(GRASP_COLOR_SELECTED);
                            selectedObject.material.emissive.setHex(GRASP_COLOR_SELECTED);
                        } else {
                            // Yellow highlight for other objects
                            selectedObject.material.color.setHex(0xffff00);
                        }
                    }
                    
                    // Add axes helper to selected object (only for component objects, not grasp points)
                    if (selectedObject.userData.type === 'component') {
                        objectAxesHelper = new THREE.AxesHelper(0.1);  // Smaller size for object axes
                        selectedObject.add(objectAxesHelper);
                    }
                    
                    updateSelectedObjectInfo();
                    updateStatus(`Selected: ${selectedObject.userData.displayName || selectedObject.userData.name}`);
                } else {
                    updateSelectedObjectInfo();
                    updateStatus("No object selected");
                }
                
                updateSceneObjectsList();
                updateStatusBar();
            }
            
            function getQuaternionDisplay(rotation) {
                // Convert Euler angles to quaternion
                const euler = new THREE.Euler(rotation.x, rotation.y, rotation.z, 'XYZ');
                const quaternion = new THREE.Quaternion();
                quaternion.setFromEuler(euler);
                
                return `X: ${quaternion.x.toFixed(4)}, Y: ${quaternion.y.toFixed(4)}, Z: ${quaternion.z.toFixed(4)}, W: ${quaternion.w.toFixed(4)}`;
            }
            
            function updateSelectedObjectInfo() {
                const container = document.getElementById('selectedObjectInfo');
                
                if (!selectedObject) {
                    container.innerHTML = '';
                    return;
                }
                
                const obj = selectedObject;
                const rot = obj.rotation;
                
                container.innerHTML = `
                    <div class="precision-controls">
                        <div class="selected-object">
                            <h4>${obj.userData.displayName || obj.userData.name}</h4>
                            <p>Type: ${obj.userData.type} | ID: ${obj.userData.id}</p>
                        </div>
                        
                        <div class="control-group">
                            <h4>Rotation (degrees)</h4>
                            <div class="control-row">
                                <span class="control-label">X:</span>
                                <input type="number" class="control-input" step="5" value="${(rot.x * 180 / Math.PI).toFixed(1)}" 
                                       onchange="setObjectRotation('x', this.value * Math.PI / 180)">
                                <div class="control-buttons">
                                    <button class="btn btn-small axis-btn x" onclick="rotateObject('x', -Math.PI / 36)">-</button>
                                    <button class="btn btn-small axis-btn x" onclick="rotateObject('x', Math.PI / 36)">+</button>
                                </div>
                            </div>
                            <div class="control-row">
                                <span class="control-label">Y:</span>
                                <input type="number" class="control-input" step="5" value="${(rot.y * 180 / Math.PI).toFixed(1)}" 
                                       onchange="setObjectRotation('y', this.value * Math.PI / 180)">
                                <div class="control-buttons">
                                    <button class="btn btn-small axis-btn y" onclick="rotateObject('y', -Math.PI / 36)">-</button>
                                    <button class="btn btn-small axis-btn y" onclick="rotateObject('y', Math.PI / 36)">+</button>
                                </div>
                            </div>
                            <div class="control-row">
                                <span class="control-label">Z:</span>
                                <input type="number" class="control-input" step="5" value="${(rot.z * 180 / Math.PI).toFixed(1)}" 
                                       onchange="setObjectRotation('z', this.value * Math.PI / 180)">
                                <div class="control-buttons">
                                    <button class="btn btn-small axis-btn z" onclick="rotateObject('z', -Math.PI / 36)">-</button>
                                    <button class="btn btn-small axis-btn z" onclick="rotateObject('z', Math.PI / 36)">+</button>
                                </div>
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <h4>Quaternion</h4>
                            <div id="quaternionDisplay" style="padding: 10px; background: #f8f9fa; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 13px;">
                                ${getQuaternionDisplay(rot)}
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <h4>Quick Actions</h4>
                            <div class="quick-actions">
                                <button class="btn btn-small" onclick="resetObjectRotation()">Reset Rotation</button>
                                <button class="btn btn-small btn-secondary" onclick="deleteSelectedObject()">Delete</button>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            
            function updateStatusBar() {
                const graspCount = graspPointsList.length;
                document.getElementById('objectCount').textContent = `Grasp Points: ${graspCount}`;
                document.getElementById('selectedInfo').textContent = 
                    selectedObject ? `Selected: ${selectedObject.userData.displayName || selectedObject.userData.name}` : 'Selected: None';
            }
            
            // Compatibility function (kept for existing code references)
            function updateSceneObjectsList() {
                // No longer needed - grasp points list removed
            }
            
            // Precision control functions
            function setObjectRotation(axis, value) {
                if (!selectedObject) return;
                selectedObject.rotation[axis] = parseFloat(value);
                updateSelectedObjectInfo();  // Update to refresh quaternion display
                updateStatus(`Rotated ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${(value * 180 / Math.PI).toFixed(1)}°`);
            }
            
            function rotateObject(axis, delta) {
                if (!selectedObject) return;
                selectedObject.rotation[axis] += delta;
                updateSelectedObjectInfo();  // Update to refresh quaternion display
                updateStatus(`Rotated ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${(selectedObject.rotation[axis] * 180 / Math.PI).toFixed(1)}°`);
            }
            
            function resetObjectRotation() {
                if (!selectedObject) return;
                selectedObject.rotation.set(0, 0, 0);
                updateSelectedObjectInfo();  // Update to refresh quaternion display
                updateStatus(`Reset rotation for ${selectedObject.userData.displayName}`);
            }
            
            function deleteSelectedObject() {
                if (!selectedObject) return;
                
                const objectName = selectedObject.userData.displayName || selectedObject.userData.name;
                
                // Remove from scene
                scene.remove(selectedObject);
                
                // Remove from objects array
                const index = sceneObjects.indexOf(selectedObject);
                if (index > -1) {
                    sceneObjects.splice(index, 1);
                }
                
                // Clear selection
                selectedObject = null;
                
                updateSceneObjectsList();
                updateSelectedObjectInfo();
                updateStatusBar();
                updateStatus(`Deleted ${objectName}`);
                showMessage(`Deleted ${objectName}`, "info");
            }
            
            function getComponentColor(componentName) {
                const colors = {
                    'base_scaled70': 0x4CAF50,
                    'fork_orange_scaled70': 0xFF9800,
                    'fork_yellow_scaled70': 0xFFEB3B,
                    'line_brown_scaled70': 0x8D6E63,
                    'line_red_scaled70': 0xF44336
                };
                return colors[componentName] || 0x2196F3;
            }
            
            function generateId() {
                return Math.random().toString(36).substr(2, 9);
            }
            
            // Utility functions
            function clearScene() {
                sceneObjects.forEach(obj => {
                    scene.remove(obj);
                });
                sceneObjects = [];
                selectedObject = null;
                currentObjectName = null;
                currentWireframe = null;
                clearGraspPoints();
                updateSelectedObjectInfo();
                updateStatusBar();
                updateStatus("Scene cleared");
                showMessage("All components removed from scene", "info");
            }
            
            function toggleGrid() {
                gridVisible = !gridVisible;
                gridHelper.visible = gridVisible;
                updateStatus(`Grid ${gridVisible ? 'shown' : 'hidden'}`);
            }
            
            function resetCamera() {
                camera.position.set(0.5, -0.5, 0.5);
                camera.up.set(0, 0, 1);
                camera.lookAt(0, 0, 0);
                controls.reset();
                updateStatus("Camera reset to default position");
            }
            
            function showFloatingControls() {
                document.getElementById('floatingControls').style.display = 'block';
                document.getElementById('showControlsBtn').style.display = 'none';
            }
            
            function hideFloatingControls() {
                document.getElementById('floatingControls').style.display = 'none';
                document.getElementById('showControlsBtn').style.display = 'block';
            }
            
            function updateStatus(message) {
                const statusBar = document.getElementById('statusBar');
                const statusSpan = statusBar.querySelector('span');
                statusSpan.textContent = message;
            }
            
            function showMessage(message, type) {
                const container = document.getElementById('statusMessages');
                const div = document.createElement('div');
                div.className = type;
                div.textContent = message;
                container.appendChild(div);
                
                // Auto-remove after 4 seconds
                setTimeout(() => {
                    if (div.parentNode) {
                        div.parentNode.removeChild(div);
                    }
                }, 4000);
            }
            
            // Grasp Points Visualization Functions
            async function loadGraspPointsForObject(objectName) {
                try {
                    updateStatus(`Loading grasp points for ${objectName}...`);
                    
                    // Load grasp points from API
                    const response = await fetch(`/api/grasp-data/${objectName}`);
                    if (!response.ok) {
                        if (response.status === 404) {
                            // No grasp points file found for this object
                            updateStatus(`No grasp points file found for ${objectName}`);
                            return;
                        }
                        throw new Error(`Failed to load grasp points: ${response.statusText}`);
                    }
                    
                    const tempData = await response.json();
                    
                    // Validate grasp points data
                    if (!tempData.grasp_points || !Array.isArray(tempData.grasp_points)) {
                        throw new Error("Invalid grasp points data - missing grasp_points array");
                    }
                    
                    // Clear previous grasp points
                    clearGraspPoints();
                    
                    // Now set the validated data
                    graspPointsData = tempData;
                    
                    // Create group for grasp points
                    graspPointsGroup = new THREE.Group();
                    graspPointsGroup.name = "GraspVisualization";
                    
                    // Add grasp points (stored relative to CAD center)
                    graspPointsList = [];
                    let totalPoints = 0;
                    graspPointsData.grasp_points.forEach((graspPoint, idx) => {
                        const sphere = createGraspPointSphere(graspPoint, idx);
                        graspPointsGroup.add(sphere);
                        graspPointsList.push(sphere);
                        totalPoints++;
                    });
                    
                    // Add grasp points group as child of wireframe so they move together
                    if (currentWireframe) {
                        currentWireframe.add(graspPointsGroup);
                    } else {
                        // Fallback: add to scene if wireframe not available
                        scene.add(graspPointsGroup);
                    }
                    
                    // Update info
                    const infoText = `Loaded ${totalPoints} grasp points`;
                    showMessage(infoText, "success");
                    updateStatus(`Grasp points loaded for ${objectName}`);
                    updateStatusBar();
                    
                } catch (error) {
                    showMessage(`Error loading grasp points: ${error.message}`, "error");
                    updateStatus("Error loading grasp points");
                    console.error("Grasp points loading error:", error);
                }
            }
            
            
            function createGraspPointSphere(graspPoint, index) {
                // Create sphere geometry for grasp point (matching grasp_candidates visualization)
                // Note: All coordinates are in meters relative to CAD center
                const geometry = new THREE.SphereGeometry(0.003, 16, 16);  // 0.003 m = 3mm
                
                // Use default green color (will change to red when selected)
                const material = new THREE.MeshPhongMaterial({
                    color: GRASP_COLOR_DEFAULT,
                    emissive: GRASP_COLOR_DEFAULT,
                    emissiveIntensity: 0.5,
                    transparent: true,
                    opacity: 0.9
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                
                // Set position from grasp point data (in meters, relative to CAD center)
                const pos = graspPoint.position;
                sphere.position.set(pos.x, pos.y, pos.z);
                
                // Store metadata
                sphere.userData = {
                    name: `Grasp-${graspPoint.id}`,
                    type: 'grasp_point',
                    graspId: graspPoint.id,
                    displayName: `Grasp Point ${graspPoint.id}`,
                    originalColor: GRASP_COLOR_DEFAULT,
                    id: generateId(),
                    isClickable: true  // Mark as clickable for selection
                };
                
                // Add gripper visualization (parallel-jaw gripper) using approach_vector from grasp point data
                if (graspPoint.approach_vector) {
                    const gripperGroup = new THREE.Group();
                    
                    // Gripper parameters
                    const gripperWidth = 0.008;      // 8mm - distance between jaws
                    const gripperLength = 0.012;     // 12mm - length of gripper fingers
                    const gripperDepth = 0.004;      // 4mm - depth of gripper fingers
                    const gripperThickness = 0.001;  // 1mm - thickness of each jaw
                    const approachLength = 0.015;    // 15mm - length of approach indicator
                    
                    // Material for gripper
                    const gripperMaterial = new THREE.MeshPhongMaterial({ 
                        color: GRIPPER_COLOR_DEFAULT,
                        transparent: true,
                        opacity: 0.8
                    });
                    
                    // Create two parallel jaw plates
                    const jawGeometry = new THREE.BoxGeometry(gripperThickness, gripperDepth, gripperLength);
                    
                    // Left jaw
                    const leftJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
                    leftJaw.position.set(-gripperWidth / 2, 0, 0);
                    gripperGroup.add(leftJaw);
                    
                    // Right jaw
                    const rightJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
                    rightJaw.position.set(gripperWidth / 2, 0, 0);
                    gripperGroup.add(rightJaw);
                    
                    // Palm/base plate connecting the jaws
                    const palmGeometry = new THREE.BoxGeometry(gripperWidth, gripperDepth * 0.6, gripperThickness);
                    const palm = new THREE.Mesh(palmGeometry, gripperMaterial);
                    palm.position.set(0, 0, -gripperLength / 2);
                    gripperGroup.add(palm);
                    
                    // Approach direction indicator (line from gripper to grasp point)
                    const approachGeometry = new THREE.CylinderGeometry(0.0008, 0.0008, approachLength, 8);
                    const approachMaterial = new THREE.MeshPhongMaterial({ 
                        color: 0xffaa00,  // Match gripper color
                        transparent: true,
                        opacity: 0.8
                    });
                    const approachLine = new THREE.Mesh(approachGeometry, approachMaterial);
                    // Rotate to be vertical (pointing down along Z axis)
                    approachLine.rotation.x = Math.PI / 2;  // Rotate 90 degrees to be vertical
                    approachLine.position.set(0, 0, -gripperLength / 2 - approachLength / 2);
                    gripperGroup.add(approachLine);
                    
                    // Orient gripper based on approach_vector
                    // Use approach vector directly (points in the direction the gripper approaches from)
                    const approach = graspPoint.approach_vector;
                    const approachVec = new THREE.Vector3(approach.x, approach.y, approach.z).normalize();
                    
                    // Default gripper points along -Z axis, rotate to match approach vector
                    const defaultDir = new THREE.Vector3(0, 0, -1);
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(defaultDir, approachVec);
                    gripperGroup.quaternion.copy(quaternion);
                    
                    // Position gripper offset from sphere along approach direction
                    // Offset distance: 8mm (0.008m) - this is just for visualization
                    const offsetDistance = 0.008; // 8mm for visualization
                    const offsetPosition = approachVec.clone().multiplyScalar(offsetDistance);
                    gripperGroup.position.copy(offsetPosition);
                    sphere.add(gripperGroup);
                    
                    // Store gripper reference in sphere userData
                    sphere.userData.gripperGroup = gripperGroup;
                    sphere.userData.gripperMaterial = gripperMaterial;
                }
                
                // Add to sceneObjects for selection
                sceneObjects.push(sphere);
                
                return sphere;
            }
            
            function clearGraspPoints() {
                if (graspPointsGroup) {
                    // Remove all grasp point objects from sceneObjects
                    graspPointsList.forEach(graspPointObj => {
                        const index = sceneObjects.indexOf(graspPointObj);
                        if (index > -1) {
                            sceneObjects.splice(index, 1);
                        }
                    });
                    
                    // Remove from parent (wireframe or scene)
                    if (graspPointsGroup.parent) {
                        graspPointsGroup.parent.remove(graspPointsGroup);
                    } else {
                        scene.remove(graspPointsGroup);
                    }
                    graspPointsGroup = null;
                }
                graspPointsData = null;
                graspPointsList = [];
                
                // Deselect if a grasp point was selected
                if (selectedObject && selectedObject.userData.type === 'grasp_point') {
                    selectedObject = null;
                }
                
                updateStatusBar();
            }
            
            // Initialize when page loads
            window.addEventListener('load', initScene);
        </script>
    </body>
    </html>
    """

@app.get("/api/components")
async def get_components():
    """Get all available components with their wireframe and ArUco data."""
    components = {}
    
    for component_name in COMPONENTS:
        try:
            # Load wireframe data
            wireframe_path = DATA_DIR / "wireframe" / f"{component_name}_wireframe.json"
            aruco_path = DATA_DIR / "aruco" / f"{component_name}_aruco.json"
            
            if wireframe_path.exists():
                with open(wireframe_path, 'r') as f:
                    wireframe_data = json.load(f)
                
                # Load ArUco data if available
                aruco_data = None
                if aruco_path.exists():
                    with open(aruco_path, 'r') as f:
                        aruco_data = json.load(f)
                
                components[component_name] = {
                    "wireframe": wireframe_data,
                    "aruco": aruco_data,
                    "name": component_name,
                    "display_name": component_name.replace('_scaled70', '').replace('_', ' ').title()
                }
            else:
                print(f"Warning: Wireframe file not found for {component_name}")
                
        except Exception as e:
            print(f"Error loading component {component_name}: {e}")
            continue
    
    return components

@app.get("/api/components/{component_name}")
async def get_component(component_name: str):
    """Get a specific component's data."""
    if component_name not in COMPONENTS:
        raise HTTPException(status_code=404, detail="Component not found")
    
    try:
        wireframe_path = DATA_DIR / "wireframe" / f"{component_name}_wireframe.json"
        aruco_path = DATA_DIR / "aruco" / f"{component_name}_aruco.json"
        
        if not wireframe_path.exists():
            raise HTTPException(status_code=404, detail="Wireframe data not found")
        
        with open(wireframe_path, 'r') as f:
            wireframe_data = json.load(f)
        
        aruco_data = None
        if aruco_path.exists():
            with open(aruco_path, 'r') as f:
                aruco_data = json.load(f)
        
        return {
            "wireframe": wireframe_data,
            "aruco": aruco_data,
            "name": component_name,
            "display_name": component_name.replace('_scaled70', '').replace('_', ' ').title()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/grasp-data/{object_name}")
async def get_grasp_data(object_name: str):
    """Get grasp points data for an object."""
    grasp_file = DATA_DIR / "grasp" / f"{object_name}_grasp_points_all_markers.json"
    
    if not grasp_file.exists():
        raise HTTPException(status_code=404, detail=f"Grasp data not found for {object_name}")
    
    try:
        with open(grasp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp data: {str(e)}")

def main():
    """Main entry point for the symmetry exporter application."""
    print("🚀 Starting Symmetry Exporter...")
    print("📱 Open your browser to: http://localhost:8002")
    print("🎯 Features:")
    print("   • Load and display wireframe components")
    print("   • Visualize grasp points")
    print("   • Interactive 3D environment")
    print("   • Precision position and rotation controls")
    print("🎮 Controls:")
    print("   • Click objects to select")
    print("   • Use control panel for precision movement")
    print("   • Arrow keys for position, Q/E for rotation")
    print("   • Mouse: wheel=zoom, right-drag=orbit")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)

if __name__ == "__main__":
    main()

