#!/usr/bin/env python3
"""
Grasp Candidates Visualization UI
Web-based 3D visualization tool for viewing objects with grasp points overlayed.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import subprocess
import os
import numpy as np

app = FastAPI(
    title="Grasp Candidates Visualizer",
    description="Interactive 3D visualization of objects with grasp points",
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
GRASP_DIR = DATA_DIR / "grasp"
WIREFRAME_DIR = DATA_DIR / "wireframe"
ARUCO_DIR = DATA_DIR / "aruco"
GRASP_CANDIDATES_DIR = DATA_DIR / "grasp_candidates"


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML interface."""
    return r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grasp Candidates Visualizer</title>
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
                width: 350px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 20px;
                overflow-y: auto;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                border-right: 1px solid #ddd;
            }
            
            .main-viewer {
                flex: 1;
                position: relative;
                background: #1a1a1a;
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
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
                color: #2c3e50;
                font-weight: 500;
                font-size: 14px;
            }
            
            .form-group select {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                background: white;
                cursor: pointer;
            }
            
            .form-group select:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            
            .btn-execute {
                width: 100%;
                padding: 12px 20px;
                background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(46, 204, 113, 0.3);
            }
            
            .btn-execute:hover:not(:disabled) {
                background: linear-gradient(135deg, #229954 0%, #27ae60 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(46, 204, 113, 0.4);
            }
            
            .btn-execute:active:not(:disabled) {
                transform: translateY(0);
                box-shadow: 0 2px 4px rgba(46, 204, 113, 0.3);
            }
            
            .btn-execute:disabled {
                background: #95a5a6;
                cursor: not-allowed;
                opacity: 0.6;
            }
            
            .btn-gripper {
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 5px;
                min-width: 120px;
            }
            
            .btn-gripper-open {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
            }
            
            .btn-gripper-open:hover {
                background: linear-gradient(135deg, #2980b9 0%, #1f6391 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4);
            }
            
            .btn-gripper-close {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                box-shadow: 0 4px 8px rgba(231, 76, 60, 0.3);
            }
            
            .btn-gripper-close:hover {
                background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(231, 76, 60, 0.4);
            }
            
            .gripper-buttons {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            
            .info-panel {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border: 1px solid #e0e0e0;
            }
            
            .info-panel h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 16px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }
            
            .info-item {
                margin-bottom: 10px;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 13px;
            }
            
            .info-item strong {
                color: #2c3e50;
                display: block;
                margin-bottom: 3px;
            }
            
            .info-item span {
                color: #666;
                font-family: 'Courier New', monospace;
            }
            
            #viewer-container {
                width: 100%;
                height: 100%;
            }
            
            .status-bar {
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 100;
            }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="sidebar">
                <div class="controls-panel">
                    <h3>📦 Object Selection</h3>
                    <div class="form-group">
                        <label for="objectSelect">Select Object:</label>
                        <select id="objectSelect">
                            <option value="">-- Select an object --</option>
                        </select>
                    </div>
                </div>
                
                <div class="controls-panel">
                    <h3>🎯 Grasp Point Selection</h3>
                    <div class="form-group">
                        <label for="graspSelect">Select Grasp Point (to filter):</label>
                        <select id="graspSelect" disabled>
                            <option value="">-- Select object first --</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="graspCandidateSelect">Select Grasp Candidate:</label>
                        <select id="graspCandidateSelect" disabled>
                            <option value="">-- Select grasp point first --</option>
                        </select>
                    </div>
                    <div class="form-group" id="executeButtonGroup" style="display: none;">
                        <button id="executeButton" class="btn-execute" disabled>Execute Grasp</button>
                    </div>
                    <div class="form-group">
                        <h3>🤖 Gripper Control</h3>
                        <div class="gripper-buttons">
                            <button id="openGripperButton" class="btn-gripper btn-gripper-open">Open Gripper</button>
                            <button id="closeGripperButton" class="btn-gripper btn-gripper-close">Close Gripper</button>
                        </div>
                    </div>
                </div>
                
                <div class="info-panel">
                    <h3>ℹ️ Grasp Point Info</h3>
                    <div id="graspInfo">
                        <div class="info-item">
                            <strong>No grasp point selected</strong>
                            <span>Select a grasp point to view details</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="main-viewer">
                <div id="viewer-container"></div>
                <div class="status-bar" id="statusBar">Ready</div>
            </div>
        </div>
        
        <script>
            // Three.js scene setup
            let scene, camera, renderer, controls;
            let graspPointsGroup = null;
            let currentObjectData = null;
            let currentGraspData = null;
            let selectedGraspId = null;  // For filtering visibility
            let selectedGraspCandidateId = null;  // For execution (direction_id)
            let selectedGraspPointIdForCandidate = null;  // Grasp point ID for the selected candidate
            
            // Colors
            const GRASP_COLOR_DEFAULT = 0x00ff00;  // Green
            const GRASP_COLOR_SELECTED = 0x3498db;  // Blue
            const WIREFRAME_COLOR = 0x888888;  // Gray
            
            function initScene() {
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                // Camera
                camera = new THREE.PerspectiveCamera(
                    75,
                    window.innerWidth / window.innerHeight,
                    0.001,
                    1000
                );
                camera.position.set(0.5, -0.5, 0.5);  // Move camera to look at scene from Y-negative
                camera.up.set(0, 0, 1);  // Set Z as up vector
                camera.lookAt(0, 0, 0);
                
                // Renderer
                const container = document.getElementById('viewer-container');
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);
                
                // Controls
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
                scene.add(directionalLight);
                
                const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
                scene.add(hemisphereLight);
                
                // Grid helper
                const gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
                gridHelper.rotateX(Math.PI / 2);  // Rotate 90 degrees to make it horizontal in XY plane
                scene.add(gridHelper);
                
                // Axes helper
                const axesHelper = new THREE.AxesHelper(0.2);
                scene.add(axesHelper);
                
                // Load available objects
                loadObjects();
                
                // Event listeners
                document.getElementById('objectSelect').addEventListener('change', onObjectChanged);
                document.getElementById('graspSelect').addEventListener('change', onGraspPointChanged);
                document.getElementById('graspCandidateSelect').addEventListener('change', onGraspCandidateChanged);
                document.getElementById('executeButton').addEventListener('click', onExecuteButtonClick);
                document.getElementById('openGripperButton').addEventListener('click', onOpenGripperClick);
                document.getElementById('closeGripperButton').addEventListener('click', onCloseGripperClick);
                
                // Window resize
                window.addEventListener('resize', onWindowResize);
                
                // Animation loop
                animate();
            }
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            
            function onWindowResize() {
                const container = document.getElementById('viewer-container');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
            
            async function loadObjects() {
                try {
                    const response = await fetch('/api/objects');
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const objects = await response.json();
                    
                    console.log('Loaded objects:', objects);
                    
                    const select = document.getElementById('objectSelect');
                    select.innerHTML = '<option value="">-- Select an object --</option>';
                    
                    if (objects && objects.length > 0) {
                        objects.forEach(objName => {
                            const option = document.createElement('option');
                            option.value = objName;
                            // Format object name: remove _scaled70, replace _ with space, capitalize words
                            const displayName = objName.replace('_scaled70', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            option.textContent = displayName;
                            select.appendChild(option);
                        });
                        
                        updateStatus(`Loaded ${objects.length} objects`);
                    } else {
                        updateStatus('No objects found');
                        console.warn('No objects returned from API');
                    }
                } catch (error) {
                    console.error('Error loading objects:', error);
                    updateStatus(`Error loading objects: ${error.message}`);
                }
            }
            
            async function onObjectChanged(event) {
                const objectName = event.target.value;
                
                if (!objectName) {
                    clearScene();
                    return;
                }
                
                try {
                    updateStatus(`Loading ${objectName}...`);
                    
                    // Load grasp data
                    const graspResponse = await fetch(`/api/grasp-data/${objectName}`);
                    if (!graspResponse.ok) {
                        throw new Error('Failed to load grasp data');
                    }
                    currentGraspData = await graspResponse.json();
                    
                    // Load wireframe
                    const wireframeResponse = await fetch(`/api/wireframe/${objectName}`);
                    if (!wireframeResponse.ok) {
                        throw new Error('Failed to load wireframe');
                    }
                    const wireframeData = await wireframeResponse.json();
                    
                    // Load ArUco data
                    const arucoResponse = await fetch(`/api/aruco/${objectName}`);
                    const arucoData = arucoResponse.ok ? await arucoResponse.json() : null;
                    
                    // Merge wireframe and ArUco markers into grasp data
                    currentGraspData.wireframe = wireframeData;
                    
                    // If grasp data doesn't have markers, use ArUco markers
                    if (!currentGraspData.markers && arucoData && arucoData.markers) {
                        // Convert ArUco markers format to grasp markers format
                        currentGraspData.markers = arucoData.markers.map(marker => ({
                            aruco_id: marker.aruco_id,
                            size: marker.size,
                            pose_absolute: marker.pose_absolute
                        }));
                    }
                    
                    // Clear previous visualization
                    clearGraspPoints();
                    
                    // Create group for everything (wireframe, markers, grasp points)
                    graspPointsGroup = new THREE.Group();
                    graspPointsGroup.name = "GraspVisualization";
                    
                    // 1. Create wireframe
                    const wireframe = createWireframeFromData(currentGraspData);
                    graspPointsGroup.add(wireframe);
                    
                    // 2. Add ArUco markers
                    if (currentGraspData.markers && Array.isArray(currentGraspData.markers)) {
                        currentGraspData.markers.forEach(markerData => {
                            const markerMesh = createMarkerMesh(markerData);
                            graspPointsGroup.add(markerMesh);
                        });
                    }
                    
                    // 3. Add grasp points (stored relative to CAD center)
                    let totalPoints = 0;
                    currentGraspData.grasp_points.forEach((graspPoint, idx) => {
                        const sphere = createGraspPointSphere(graspPoint, idx);
                        graspPointsGroup.add(sphere);
                        totalPoints++;
                    });
                    
                    // Objects are always at world center (0,0,0) - no pose transformation
                    // Grasp points are relative to CAD center which is at world center
                    graspPointsGroup.position.set(0, 0, 0);
                    graspPointsGroup.rotation.set(0, 0, 0);
                    
                    scene.add(graspPointsGroup);
                    
                    // Update grasp point selector
                    updateGraspPointSelector();
                    
                    updateStatus(`Loaded ${objectName}`);
                } catch (error) {
                    console.error('Error loading object:', error);
                    updateStatus(`Error loading ${objectName}: ${error.message}`);
                }
            }
            
            function createWireframeFromData(data) {
                const vertices = data.wireframe.vertices;
                const edges = data.wireframe.edges;
                
                const lineVertices = [];
                edges.forEach(edge => {
                    const v1 = vertices[edge[0]];
                    const v2 = vertices[edge[1]];
                    lineVertices.push(...v1, ...v2);
                });
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
                
                const material = new THREE.LineBasicMaterial({ 
                    color: 0x4a90e2,
                    linewidth: 2
                });
                
                const wireframe = new THREE.LineSegments(geometry, material);
                wireframe.userData = { 
                    name: data.object_name,
                    type: 'grasp_wireframe',
                    displayName: data.display_name,
                    originalColor: 0x4a90e2,
                    id: generateId()
                };
                
                // Center the wireframe
                wireframe.position.set(0, 0, 0);
                
                return wireframe;
            }
            
            function createMarkerMesh(markerData) {
                const size = markerData.size;
                const geometry = new THREE.BoxGeometry(size, size, size * 0.1);
                const material = new THREE.MeshBasicMaterial({ 
                    color: 0xff6b6b,
                    transparent: true,
                    opacity: 0.7
                });
                
                const marker = new THREE.Mesh(geometry, material);
                
                const pos = markerData.pose_absolute.position;
                marker.position.set(pos.x, pos.y, pos.z);
                
                const rot = markerData.pose_absolute.rotation;
                marker.rotation.set(rot.roll, rot.pitch, rot.yaw);
                
                marker.userData = { 
                    name: `ArUco-${markerData.aruco_id}`,
                    type: 'grasp_marker',
                    arucoId: markerData.aruco_id,
                    displayName: `ArUco ${markerData.aruco_id}`,
                    originalColor: 0xff6b6b,
                    id: generateId()
                };
                
                return marker;
            }
            
            function createGraspPointSphere(graspPoint, index) {
                // Create sphere geometry for grasp point
                // Note: All coordinates are in meters relative to CAD center
                const geometry = new THREE.SphereGeometry(0.003, 16, 16);  // 0.003 m = 3mm
                const isSelected = selectedGraspId === graspPoint.id;
                const color = isSelected ? GRASP_COLOR_SELECTED : GRASP_COLOR_DEFAULT;
                const material = new THREE.MeshPhongMaterial({
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.5,
                    transparent: true,
                    opacity: 0.9
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                
                // Set position from grasp point data (in meters, relative to CAD center)
                // Coordinate flip already applied during export
                const pos = graspPoint.position;
                sphere.position.set(pos.x, pos.y, pos.z);
                
                // Store metadata
                sphere.userData = {
                    name: `Grasp-${graspPoint.id}`,
                    type: 'grasp_point',
                    graspId: graspPoint.id,
                    displayName: `Grasp Point ${graspPoint.id}`,
                    originalColor: color,
                    id: generateId()
                };
                
                // Add gripper visualization (parallel-jaw gripper)
                const gripperGroup = new THREE.Group();
                
                // Gripper parameters
                const gripperWidth = 0.008;      // 8mm - distance between jaws
                const gripperLength = 0.012;     // 12mm - length of gripper fingers
                const gripperDepth = 0.004;      // 4mm - depth of gripper fingers
                const gripperThickness = 0.001;  // 1mm - thickness of each jaw
                const approachLength = 0.015;    // 15mm - length of approach indicator
                
                // Material for gripper
                const gripperMaterial = new THREE.MeshPhongMaterial({ 
                    color: 0xffaa00,
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
                const approachGeometry = new THREE.CylinderGeometry(0.0008, 0.0008, approachLength, 8);  // Increased width to 0.8mm
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
                
                // Orient gripper based on approach vector if available
                if (graspPoint.approach_vector) {
                    const approach = graspPoint.approach_vector;
                    const approachVec = new THREE.Vector3(approach.x, approach.y, approach.z).normalize();
                    
                    // Default gripper points along -Z axis, rotate to match approach vector
                    const defaultDir = new THREE.Vector3(0, 0, -1);
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(defaultDir, approachVec);
                    gripperGroup.quaternion.copy(quaternion);
                }
                
                // Position gripper slightly offset from sphere along approach direction
                gripperGroup.position.set(0, 0, 0.010);  // 10mm offset
                sphere.add(gripperGroup);
                
                // Store gripper reference in sphere userData for visibility control
                sphere.userData.gripperGroup = gripperGroup;
                
                // Make gripper group and all its children clickable
                gripperGroup.userData.isGripper = true;
                gripperGroup.traverse((child) => {
                    child.userData.isGripper = true;
                });
                
                return sphere;
            }
            
            function clearGraspPoints() {
                if (graspPointsGroup) {
                    scene.remove(graspPointsGroup);
                    graspPointsGroup = null;
                }
            }
            
            function updateGraspPointSelector() {
                const select = document.getElementById('graspSelect');
                select.innerHTML = '<option value="">-- Select a grasp point --</option>';
                
                if (!currentGraspData || !currentGraspData.grasp_points) {
                    select.disabled = true;
                    return;
                }
                
                select.disabled = false;
                
                currentGraspData.grasp_points.forEach(gp => {
                    const option = document.createElement('option');
                    option.value = gp.id.toString();
                    option.textContent = `Grasp Point ${gp.id}`;
                    select.appendChild(option);
                });
            }
            
            function onGraspPointChanged(event) {
                const graspId = event.target.value;
                
                if (!graspId) {
                    selectedGraspId = null;
                    // Re-render grasp points to update colors and show all grippers
                    if (currentGraspData && graspPointsGroup) {
                        graspPointsGroup.traverse((child) => {
                            if (child.userData && child.userData.type === 'grasp_point') {
                                const isSelected = false;
                                const color = GRASP_COLOR_DEFAULT;
                                if (child.material) {
                                    child.material.color.setHex(color);
                                    child.material.emissive.setHex(color);
                                }
                                // Show all grippers when nothing is selected
                                if (child.userData.gripperGroup) {
                                    child.userData.gripperGroup.visible = true;
                                }
                            }
                        });
                    }
                    updateGraspInfo(null);
                    // Clear grasp candidate selection
                    selectedGraspCandidateId = null;
                    const graspCandidateSelect = document.getElementById('graspCandidateSelect');
                    if (graspCandidateSelect) {
                        graspCandidateSelect.innerHTML = '<option value="">-- Select grasp point first --</option>';
                        graspCandidateSelect.disabled = true;
                    }
                    // Hide execute button
                    const executeButtonGroup = document.getElementById('executeButtonGroup');
                    const executeButton = document.getElementById('executeButton');
                    if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                    if (executeButton) executeButton.disabled = true;
                    return;
                }
                
                selectedGraspId = parseInt(graspId);
                
                // Find grasp point data
                const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === selectedGraspId);
                
                if (graspPoint && graspPointsGroup) {
                    updateGraspInfo(graspPoint);
                    
                    // Save grasp candidate data automatically when grasp point is selected
                    saveGraspCandidateData(graspPoint);
                    
                    // Update colors of all grasp points and hide/show grippers
                    graspPointsGroup.traverse((child) => {
                        if (child.userData && child.userData.type === 'grasp_point') {
                            const isSelected = child.userData.graspId === selectedGraspId;
                            const color = isSelected ? GRASP_COLOR_SELECTED : GRASP_COLOR_DEFAULT;
                            if (child.material) {
                                child.material.color.setHex(color);
                                child.material.emissive.setHex(color);
                            }
                            // Hide grippers for non-selected grasp points, show for selected
                            if (child.userData.gripperGroup) {
                                child.userData.gripperGroup.visible = isSelected;
                            }
                        }
                    });
                    
                    // Populate grasp candidate dropdown with direction_id = 1 (can be extended for multiple candidates)
                    const graspCandidateSelect = document.getElementById('graspCandidateSelect');
                    if (graspCandidateSelect) {
                        graspCandidateSelect.innerHTML = '<option value="">-- Select grasp candidate --</option>';
                        const option = document.createElement('option');
                        option.value = '1';  // direction_id = 1
                        option.textContent = `Grasp Candidate 1`;
                        option.dataset.graspPointId = selectedGraspId;  // Store grasp point ID in data attribute
                        graspCandidateSelect.appendChild(option);
                        graspCandidateSelect.disabled = false;
                    }
                    
                    // Clear previous grasp candidate selection
                    selectedGraspCandidateId = null;
                    selectedGraspPointIdForCandidate = null;
                    
                    // Hide execute button until grasp candidate is selected
                    const executeButtonGroup = document.getElementById('executeButtonGroup');
                    const executeButton = document.getElementById('executeButton');
                    if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                    if (executeButton) executeButton.disabled = true;
                }
            }
            
            function onGraspCandidateChanged(event) {
                const selectedOption = event.target.options[event.target.selectedIndex];
                const directionId = event.target.value;
                
                if (!directionId) {
                    selectedGraspCandidateId = null;
                    selectedGraspPointIdForCandidate = null;
                    // Hide execute button
                    const executeButtonGroup = document.getElementById('executeButtonGroup');
                    const executeButton = document.getElementById('executeButton');
                    if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                    if (executeButton) executeButton.disabled = true;
                    return;
                }
                
                selectedGraspCandidateId = parseInt(directionId);  // direction_id (e.g., 1)
                selectedGraspPointIdForCandidate = parseInt(selectedOption.dataset.graspPointId);  // grasp point ID
                
                // Load and visualize grasp candidate data
                loadAndVisualizeGraspCandidate(selectedGraspPointIdForCandidate, selectedGraspCandidateId);
                
                // Show and enable execute button
                const executeButtonGroup = document.getElementById('executeButtonGroup');
                const executeButton = document.getElementById('executeButton');
                if (executeButtonGroup) executeButtonGroup.style.display = 'block';
                if (executeButton) executeButton.disabled = false;
            }
            
            async function saveGraspCandidateData(graspPoint) {
                try {
                    const approach = graspPoint.approach_vector || { x: 0, y: 0, z: 1 };
                    const response = await fetch('/api/save-grasp-candidate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            object_name: currentObjectData.object_name,
                            grasp_point_id: graspPoint.id,
                            direction_id: 1,
                            grasp_point_position: graspPoint.position,
                            approach_vector: approach
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        console.log('Grasp candidate data saved:', result);
                    }
                } catch (error) {
                    console.error('Error saving grasp candidate data:', error);
                }
            }
            
            async function loadAndVisualizeGraspCandidate(graspPointId, directionId) {
                try {
                    const response = await fetch(`/api/grasp-candidate/${currentObjectData.object_name}/${graspPointId}/${directionId}`);
                    
                    if (!response.ok) {
                        console.warn('Grasp candidate data not found, using default visualization');
                        return;
                    }
                    
                    const candidateData = await response.json();
                    
                    // Update gripper position to grasp candidate position
                    if (graspPointsGroup) {
                        graspPointsGroup.traverse((child) => {
                            if (child.userData && child.userData.type === 'grasp_point' && 
                                child.userData.graspId === graspPointId) {
                                // Move gripper to grasp candidate position
                                if (child.userData.gripperGroup) {
                                    const candidatePos = candidateData.grasp_candidate_position;
                                    const graspPointPos = candidateData.grasp_point_position;
                                    
                                    // Calculate offset from grasp point to candidate
                                    const offset = new THREE.Vector3(
                                        candidatePos.x - graspPointPos.x,
                                        candidatePos.y - graspPointPos.y,
                                        candidatePos.z - graspPointPos.z
                                    );
                                    
                                    // Position gripper at candidate position (relative to sphere)
                                    child.userData.gripperGroup.position.copy(offset);
                                    child.userData.gripperGroup.visible = true;
                                }
                            }
                        });
                    }
                } catch (error) {
                    console.error('Error loading grasp candidate data:', error);
                }
            }
            
            function generateId() {
                return Math.random().toString(36).substr(2, 9);
            }
            
            function onExecuteButtonClick() {
                // Execute grasp when button is clicked
                // Use grasp point ID for execution (the robot uses grasp point ID, not direction_id)
                if (selectedGraspPointIdForCandidate && currentGraspData) {
                    executeGrasp(currentGraspData.object_name, selectedGraspPointIdForCandidate);
                }
            }
            
            async function executeGrasp(objectName, graspId) {
                try {
                    updateStatus(`Executing grasp for ${objectName} at grasp point ${graspId}...`);
                    
                    const response = await fetch('/api/execute-grasp', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            object_name: objectName,
                            grasp_id: graspId,
                            topic: '/objects_poses_sim',
                            movement_duration: 10.0
                        })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log('Grasp execution started:', result);
                } catch (error) {
                    console.error('Error executing grasp:', error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            function onOpenGripperClick() {
                controlGripper('open');
            }
            
            function onCloseGripperClick() {
                controlGripper('close');
            }
            
            async function controlGripper(command) {
                try {
                    updateStatus(`${command === 'open' ? 'Opening' : 'Closing'} gripper...`);
                    const response = await fetch('/api/gripper-command', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ command: command })
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    updateStatus(`✅ ${result.message}`);
                    console.log(`Gripper ${command} command sent:`, result);
                } catch (error) {
                    console.error(`Error ${command === 'open' ? 'opening' : 'closing'} gripper:`, error);
                    updateStatus(`❌ Error: ${error.message}`);
                }
            }
            
            function updateGraspInfo(graspPoint) {
                const infoDiv = document.getElementById('graspInfo');
                
                if (!graspPoint) {
                    infoDiv.innerHTML = `
                        <div class="info-item">
                            <strong>No grasp point selected</strong>
                            <span>Select a grasp point to view details</span>
                        </div>
                    `;
                    return;
                }
                
                const pos = graspPoint.position;
                const approach = graspPoint.approach_vector || { x: 0, y: 0, z: 1 };
                
                infoDiv.innerHTML = `
                    <div class="info-item">
                        <strong>Grasp Point ID:</strong>
                        <span>${graspPoint.id}</span>
                    </div>
                    <div class="info-item">
                        <strong>Position:</strong>
                        <span>X: ${pos.x.toFixed(4)}m, Y: ${pos.y.toFixed(4)}m, Z: ${pos.z.toFixed(4)}m</span>
                    </div>
                    <div class="info-item">
                        <strong>Approach Vector:</strong>
                        <span>X: ${approach.x.toFixed(3)}, Y: ${approach.y.toFixed(3)}, Z: ${approach.z.toFixed(3)}</span>
                    </div>
                    <div class="info-item">
                        <strong>Type:</strong>
                        <span>${graspPoint.type || 'center_point'}</span>
                    </div>
                `;
            }
            
            function clearScene() {
                clearGraspPoints();
                currentObjectData = null;
                currentGraspData = null;
                selectedGraspId = null;
                selectedGraspCandidateId = null;
                selectedGraspPointIdForCandidate = null;
                
                document.getElementById('graspSelect').innerHTML = '<option value="">-- Select object first --</option>';
                document.getElementById('graspSelect').disabled = true;
                
                const graspCandidateSelect = document.getElementById('graspCandidateSelect');
                if (graspCandidateSelect) {
                    graspCandidateSelect.innerHTML = '<option value="">-- Select grasp point first --</option>';
                    graspCandidateSelect.disabled = true;
                }
                
                // Hide execute button
                const executeButtonGroup = document.getElementById('executeButtonGroup');
                const executeButton = document.getElementById('executeButton');
                if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                if (executeButton) executeButton.disabled = true;
                updateGraspInfo(null);
            }
            
            function updateStatus(message) {
                document.getElementById('statusBar').textContent = message;
            }
            
            // Initialize when page loads
            window.addEventListener('load', initScene);
        </script>
    </body>
    </html>
    """


@app.get("/api/objects")
async def get_objects():
    """List available objects from grasp JSON files."""
    objects = []
    
    if not GRASP_DIR.exists():
        return objects
    
    for grasp_file in GRASP_DIR.glob("*_grasp_points_all_markers.json"):
        # Extract object name from filename
        # e.g., "fork_orange_scaled70_grasp_points_all_markers.json" -> "fork_orange_scaled70"
        object_name = grasp_file.stem.replace("_grasp_points_all_markers", "")
        objects.append(object_name)
    
    return sorted(objects)


@app.get("/api/grasp-data/{object_name}")
async def get_grasp_data(object_name: str):
    """Get grasp points data for an object."""
    grasp_file = GRASP_DIR / f"{object_name}_grasp_points_all_markers.json"
    
    if not grasp_file.exists():
        raise HTTPException(status_code=404, detail=f"Grasp data not found for {object_name}")
    
    try:
        with open(grasp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp data: {str(e)}")


@app.get("/api/aruco/{object_name}")
async def get_aruco_data(object_name: str):
    """Get ArUco marker data for an object."""
    aruco_file = ARUCO_DIR / f"{object_name}_aruco.json"
    
    if not aruco_file.exists():
        return JSONResponse(
            content={"error": f"ArUco data not found for {object_name}"},
            status_code=404
        )
    
    try:
        with open(aruco_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ArUco data: {str(e)}")


@app.get("/api/wireframe/{object_name}")
async def get_wireframe(object_name: str):
    """Get wireframe data for object visualization."""
    wireframe_file = WIREFRAME_DIR / f"{object_name}_wireframe.json"
    
    if not wireframe_file.exists():
        raise HTTPException(status_code=404, detail=f"Wireframe data not found for {object_name}")
    
    try:
        with open(wireframe_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading wireframe data: {str(e)}")


class ExecuteGraspRequest(BaseModel):
    object_name: str
    grasp_id: int
    topic: str = '/objects_poses_sim'
    movement_duration: float = 10.0


@app.post("/api/execute-grasp")
async def execute_grasp(request: ExecuteGraspRequest):
    """Execute visual servo grasp for the specified object and grasp point."""
    try:
        object_name = request.object_name
        grasp_id = request.grasp_id
        topic = request.topic
        movement_duration = request.movement_duration
        
        if not object_name or grasp_id is None:
            raise HTTPException(status_code=400, detail="object_name and grasp_id are required")
        
        # Get the script path
        script_dir = Path(__file__).parent
        script_path = script_dir / "visual_servo_grasp.py"
        
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")
        
        # Map object name from JSON format (fork_yellow_scaled70) to topic format (fork_yellow)
        # Remove _scaled70 suffix if present
        topic_object_name = object_name.replace('_scaled70', '')
        
        # Build command with bash to source ROS2 and use python3.10
        # Use bash -c to source ROS2 setup and then run the script
        cmd_str = (
            f"source /opt/ros/humble/setup.bash && "
            f"python3.10 {script_path} "
            f"--object-name {topic_object_name} "
            f"--grasp-id {grasp_id} "
            f"--topic {topic} "
            f"--movement-duration {movement_duration}"
        )
        
        # Execute in background (non-blocking) using bash
        print(f"🔧 Executing command: {cmd_str}")
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            cwd=str(script_dir.parent.parent),
            shell=True,
            executable='/bin/bash',
            text=True,  # Return strings instead of bytes
            bufsize=1  # Line buffered
        )
        
        # Start a thread to read and log output
        import threading
        
        def log_output(pipe, process_pid):
            """Read output from process and log it."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line = line.strip()
                        if line:
                            print(f"[PID {process_pid}] {line}")
                pipe.close()
            except Exception as e:
                print(f"Error reading process output: {e}")
        
        # Start thread to log output
        output_thread = threading.Thread(
            target=log_output,
            args=(process.stdout, process.pid),
            daemon=True
        )
        output_thread.start()
        
        # Wait a moment to check if process started successfully
        import time
        time.sleep(0.5)
        
        # Check if process is still running or if it exited with error
        return_code = process.poll()
        if return_code is not None:
            # Process exited immediately - read any error output
            error_output = ""
            try:
                if process.stdout:
                    remaining = process.stdout.read()
                    if remaining:
                        error_output = remaining.strip()
            except:
                pass
            
            error_msg = f"Process exited immediately with code {return_code}"
            if error_output:
                error_msg += f". Output: {error_output[:500]}"  # Limit error message length
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Process started successfully with PID: {process.pid}")
        
        return JSONResponse(content={
            "status": "started",
            "message": f"Executing grasp for {topic_object_name} at grasp point {grasp_id}",
            "pid": process.pid
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing grasp: {str(e)}")


class GripperCommandRequest(BaseModel):
    command: str  # 'open' or 'close'


@app.post("/api/gripper-command")
async def gripper_command(request: GripperCommandRequest):
    """Send gripper command (open/close) via ROS2 topic."""
    try:
        command = request.command.lower()
        
        if command not in ['open', 'close']:
            raise HTTPException(status_code=400, detail="Command must be 'open' or 'close'")
        
        # Build ROS2 topic pub command
        cmd_str = (
            f"source /opt/ros/humble/setup.bash && "
            f"ros2 topic pub --once /gripper_command std_msgs/String \"{{data: '{command}'}}\""
        )
        
        # Execute command
        print(f"🔧 Executing gripper command: {cmd_str}")
        result = subprocess.run(
            cmd_str,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=5.0
        )
        
        if result.returncode != 0:
            error_msg = f"Gripper command failed with code {result.returncode}"
            if result.stderr:
                error_msg += f". Error: {result.stderr[:200]}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"✅ Gripper command '{command}' sent successfully")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Gripper {command} command sent successfully",
            "command": command
        })
        
    except subprocess.TimeoutExpired:
        error_msg = "Gripper command timed out"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error sending gripper command: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


class GraspCandidateRequest(BaseModel):
    object_name: str
    grasp_point_id: int
    direction_id: int = 1
    grasp_point_position: Dict[str, float]
    approach_vector: Dict[str, float]


@app.post("/api/save-grasp-candidate")
async def save_grasp_candidate(request: GraspCandidateRequest):
    """Save grasp candidate data (grasp point position, candidate position + 10mm, orientation)."""
    try:
        # Ensure directory exists
        GRASP_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Calculate grasp candidate position (grasp point position + 10mm along approach direction)
        approach_vec = np.array([
            request.approach_vector.get('x', 0.0),
            request.approach_vector.get('y', 0.0),
            request.approach_vector.get('z', 1.0)
        ])
        approach_vec = approach_vec / np.linalg.norm(approach_vec)  # Normalize
        
        offset_mm = 0.010  # 10mm in meters
        grasp_point_pos = np.array([
            request.grasp_point_position['x'],
            request.grasp_point_position['y'],
            request.grasp_point_position['z']
        ])
        
        grasp_candidate_position = grasp_point_pos + approach_vec * offset_mm
        
        # Create grasp candidate data structure
        grasp_candidate_data = {
            "object_name": request.object_name,
            "grasp_point_id": request.grasp_point_id,
            "direction_id": request.direction_id,
            "grasp_point_position": {
                "x": float(grasp_point_pos[0]),
                "y": float(grasp_point_pos[1]),
                "z": float(grasp_point_pos[2])
            },
            "grasp_candidate_position": {
                "x": float(grasp_candidate_position[0]),
                "y": float(grasp_candidate_position[1]),
                "z": float(grasp_candidate_position[2])
            },
            "approach_vector": {
                "x": float(approach_vec[0]),
                "y": float(approach_vec[1]),
                "z": float(approach_vec[2])
            },
            "offset_mm": 10.0
        }
        
        # Save to JSON file
        filename = f"{request.object_name}_grasp_point_{request.grasp_point_id}_direction_{request.direction_id}.json"
        file_path = GRASP_CANDIDATES_DIR / filename
        
        with open(file_path, 'w') as f:
            json.dump(grasp_candidate_data, f, indent=2)
        
        print(f"✅ Saved grasp candidate data to: {file_path}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Grasp candidate data saved for {request.object_name} grasp point {request.grasp_point_id}",
            "file_path": str(file_path),
            "data": grasp_candidate_data
        })
        
    except Exception as e:
        error_msg = f"Error saving grasp candidate: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/grasp-candidate/{object_name}/{grasp_point_id}/{direction_id}")
async def get_grasp_candidate(object_name: str, grasp_point_id: int, direction_id: int = 1):
    """Get grasp candidate data for a specific grasp point and direction."""
    filename = f"{object_name}_grasp_point_{grasp_point_id}_direction_{direction_id}.json"
    file_path = GRASP_CANDIDATES_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Grasp candidate data not found for {object_name} grasp point {grasp_point_id} direction {direction_id}"
        )
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grasp candidate data: {str(e)}")


def main():
    """Main entry point."""
    print("🚀 Starting Grasp Candidates Visualizer...")
    print("📱 Open your browser to: http://localhost:8002")
    print("🎯 Features:")
    print("   • Visualize objects with grasp points overlayed")
    print("   • Select objects and grasp points")
    print("   • Interactive 3D viewer with full rotation/zoom/pan")
    print("   • View grasp point details and positions")
    print("🎮 Controls:")
    print("   • Mouse wheel: zoom")
    print("   • Right-click + drag: rotate")
    print("   • Middle-click + drag: pan")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)


if __name__ == "__main__":
    main()

