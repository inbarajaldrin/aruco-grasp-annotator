/**
 * Enhanced 3D Assembly App - Main JavaScript
 * Interactive assembly tool with precision controls
 */

// Global variables
let scene, camera, renderer, controls;
let loadedComponents = {};
let sceneObjects = [];
let selectedObject = null;
let gridVisible = true;
let gridHelper, axesHelper;
let graspPointsGroup = null;
let graspPointsData = null;

// Initialize the 3D scene
function initScene() {
    const container = document.getElementById('viewer');

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera setup - Orthographic for consistent dimensions when zooming
    const frustumSize = 0.5;
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.OrthographicCamera(
        frustumSize * aspect / -2,
        frustumSize * aspect / 2,
        frustumSize / 2,
        frustumSize / -2,
        0.001,
        1000
    );
    camera.position.set(0.5, -0.5, 0.5);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateProjectionMatrix();

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.minZoom = 0.1;
    controls.maxZoom = 50;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, -10, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
    scene.add(hemisphereLight);

    // Grid - rotate to make it horizontal in XY plane (Z-up)
    gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
    gridHelper.rotateX(Math.PI / 2);
    scene.add(gridHelper);

    // Custom coordinate frame matching Blender convention (Z-up)
    const axesLength = 0.2;
    const axesGeometry = new THREE.BufferGeometry();
    const axesMaterial = new THREE.LineBasicMaterial({ vertexColors: true });

    const axesVertices = new Float32Array([
        0, 0, 0, axesLength, 0, 0,
        0, 0, 0, 0, axesLength, 0,
        0, 0, 0, 0, 0, axesLength
    ]);

    const axesColors = new Float32Array([
        1, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 1
    ]);

    axesGeometry.setAttribute('position', new THREE.BufferAttribute(axesVertices, 3));
    axesGeometry.setAttribute('color', new THREE.BufferAttribute(axesColors, 3));
    axesHelper = new THREE.LineSegments(axesGeometry, axesMaterial);
    scene.add(axesHelper);

    setupEventListeners();
    animate();
    updateStatus("Scene initialized - Ready to load components");
}

function setupEventListeners() {
    const canvas = renderer.domElement;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('resize', onWindowResize);
    document.addEventListener('keydown', onKeyDown);
}

function onKeyDown(event) {
    if (!selectedObject) return;

    const step = event.shiftKey ? 0.1 : 0.01;
    const rotStep = event.shiftKey ? Math.PI / 18 : Math.PI / 36;

    switch(event.key) {
        case 'ArrowLeft':
            selectedObject.position.x -= step;
            break;
        case 'ArrowRight':
            selectedObject.position.x += step;
            break;
        case 'ArrowUp':
            selectedObject.position.y += step;
            break;
        case 'ArrowDown':
            selectedObject.position.y -= step;
            break;
        case 'PageUp':
            selectedObject.position.z += step;
            break;
        case 'PageDown':
            selectedObject.position.z -= step;
            break;
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

    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'q', 'e', 'Q', 'E'].includes(event.key)) {
        event.preventDefault();
        updateSelectedObjectInfo();
    }
}

function onWindowResize() {
    const container = document.getElementById('viewer');
    const frustumSize = 0.5;
    const aspect = container.clientWidth / container.clientHeight;
    camera.left = frustumSize * aspect / -2;
    camera.right = frustumSize * aspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = frustumSize / -2;
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

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const components = await response.json();

        if (Object.keys(components).length === 0) {
            throw new Error("No components loaded. Check server logs for errors.");
        }

        loadedComponents = components;
        updateComponentList();
        showMessage(`Loaded ${Object.keys(components).length} components successfully!`, "success");
        updateStatus("All components loaded - Click on component names to add to scene");
    } catch (error) {
        console.error("Error loading components:", error);
        showMessage(`Error loading components: ${error.message}`, "error");
        updateStatus("Error loading components - Check console for details");
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

    const existingObject = sceneObjects.find(obj =>
        obj.userData &&
        obj.userData.type === 'component' &&
        obj.userData.name === componentName
    );

    if (existingObject) {
        selectObject(existingObject);
        updateStatus(`Selected existing ${component.display_name} in scene`);
        showMessage(`Selected ${component.display_name} (already in scene)`, "info");
        return;
    }

    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(component.wireframe.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    const edges = component.wireframe.edges;
    const lineGeometry = new THREE.BufferGeometry();
    const lineVertices = [];

    edges.forEach(edge => {
        const v1 = component.wireframe.vertices[edge[0]];
        const v2 = component.wireframe.vertices[edge[1]];
        lineVertices.push(...v1, ...v2);
    });

    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));

    const material = new THREE.LineBasicMaterial({
        color: getComponentColor(componentName),
        linewidth: 1
    });

    const wireframe = new THREE.LineSegments(lineGeometry, material);
    wireframe.userData = {
        name: componentName,
        type: 'component',
        displayName: component.display_name,
        originalColor: getComponentColor(componentName),
        id: generateId(),
        assemblyType: 'object',
        assemblySubtype: 'block',
        pegAxis: 'y'
    };

    wireframe.position.set(0, 0, 0);

    scene.add(wireframe);
    sceneObjects.push(wireframe);

    if (component.aruco && component.aruco.markers) {
        component.aruco.markers.forEach((marker, index) => {
            addArUcoMarker(marker, wireframe, index);
        });
    }

    updateSceneObjectsList();
    updateStatus(`Added ${component.display_name} to scene`);
    showMessage(`Added ${component.display_name} to assembly`, "success");

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

    const pos = markerData.pose_absolute.position;
    marker.position.set(pos.x, pos.y, pos.z);

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

    parentObject.add(marker);
    sceneObjects.push(marker);
}

function selectObject(object) {
    if (selectedObject && selectedObject.material) {
        selectedObject.material.color.setHex(selectedObject.userData.originalColor);
    }

    selectedObject = object;

    if (selectedObject) {
        if (selectedObject.material) {
            selectedObject.material.color.setHex(0xffff00);
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

function toggleObjectVisibility(objectId, event) {
    event.stopPropagation();
    const targetObject = sceneObjects.find(obj =>
        obj.userData && obj.userData.id === objectId
    );

    if (targetObject) {
        targetObject.visible = !targetObject.visible;
        targetObject.traverse((child) => {
            child.visible = targetObject.visible;
        });

        updateSceneObjectsList();
        updateSelectedObjectInfo();
    }
}
window.toggleObjectVisibility = toggleObjectVisibility;

function getSceneObjectsListHTML() {
    const componentObjects = sceneObjects.filter(obj =>
        obj.userData && obj.userData.type === 'component'
    );

    if (componentObjects.length === 0) {
        return '<div class="loading" style="padding: 10px; font-size: 12px; color: #666;">No objects in scene</div>';
    }

    let html = '<div class="scene-objects" style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; background: white;">';

    componentObjects.forEach(obj => {
        const isSelected = obj === selectedObject;
        const isVisible = obj.visible !== false;
        const eyeIcon = isVisible ? '👁️' : '👁️‍🗨️';
        const eyeStyle = isVisible ? 'opacity: 1;' : 'opacity: 0.4;';
        html += `
            <div class="scene-object-item ${isSelected ? 'selected' : ''}"
                 onclick="selectObjectFromTransformPanel(this)"
                 data-object-id="${obj.userData.id}"
                 style="cursor: pointer; padding: 8px 12px; border-bottom: 1px solid #eee; transition: all 0.2s; display: flex; align-items: center; justify-content: space-between; ${isSelected ? 'background: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db;' : ''}">
                <div style="display: flex; align-items: center; gap: 8px; flex: 1;">
                    <span class="object-name" style="font-weight: 500; color: #2c3e50;">${obj.userData.displayName || obj.userData.name}</span>
                    <span class="object-type" style="font-size: 11px; padding: 2px 6px; border-radius: 8px; background: #ecf0f1; color: #7f8c8d;">${obj.userData.type}</span>
                </div>
                <button onclick="toggleObjectVisibility('${obj.userData.id}', event)"
                        style="background: none; border: none; cursor: pointer; font-size: 16px; padding: 4px 8px; ${eyeStyle} transition: opacity 0.2s; z-index: 10;"
                        title="${isVisible ? 'Hide object' : 'Show object'}"
                        onmouseover="this.style.opacity='1'"
                        onmouseout="this.style.opacity='${isVisible ? '1' : '0.4'}'">
                    ${eyeIcon}
                </button>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

function getArUcoMarkersListHTML() {
    const markerObjects = sceneObjects.filter(obj => {
        if (!obj.userData || obj.userData.type !== 'marker') {
            return false;
        }
        const parentId = obj.userData.parentId;
        if (parentId) {
            const parentObject = sceneObjects.find(p =>
                p.userData && p.userData.id === parentId
            );
            return parentObject && parentObject.visible !== false;
        }
        return true;
    });

    if (markerObjects.length === 0) {
        return '<div class="loading" style="padding: 10px; font-size: 12px; color: #666;">No ArUco markers for visible objects</div>';
    }

    let html = '<div class="scene-objects" style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; background: white;">';

    markerObjects.forEach(obj => {
        const isSelected = obj === selectedObject;
        const markerName = obj.userData.displayName || `ArUco ${obj.userData.arucoId || 'Marker'}`;
        html += `
            <div class="scene-object-item ${isSelected ? 'selected' : ''}"
                 onclick="selectObjectFromTransformPanel(this)"
                 data-object-id="${obj.userData.id}"
                 style="cursor: pointer; padding: 8px 12px; border-bottom: 1px solid #eee; transition: all 0.2s; ${isSelected ? 'background: rgba(52, 152, 219, 0.1); border-left: 3px solid #3498db;' : ''}">
                <span class="object-name" style="font-weight: 500; color: #2c3e50;">${markerName}</span>
                <span class="object-type" style="font-size: 11px; padding: 2px 6px; border-radius: 8px; background: #ecf0f1; color: #7f8c8d;">${obj.userData.type}</span>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

function selectObjectFromTransformPanel(element) {
    const objectId = element.getAttribute('data-object-id');
    const targetObject = sceneObjects.find(obj =>
        obj.userData && obj.userData.id === objectId
    );

    if (targetObject) {
        selectObject(targetObject);
    }
}
window.selectObjectFromTransformPanel = selectObjectFromTransformPanel;

function calculateQuaternion(rot) {
    const c1 = Math.cos(rot.x / 2);
    const c2 = Math.cos(rot.y / 2);
    const c3 = Math.cos(rot.z / 2);
    const s1 = Math.sin(rot.x / 2);
    const s2 = Math.sin(rot.y / 2);
    const s3 = Math.sin(rot.z / 2);
    return {
        x: s1 * c2 * c3 + c1 * s2 * s3,
        y: c1 * s2 * c3 - s1 * c2 * s3,
        z: c1 * c2 * s3 + s1 * s2 * c3,
        w: c1 * c2 * c3 - s1 * s2 * s3
    };
}

function updateSelectedObjectInfo() {
    const container = document.getElementById('selectedObjectInfo');

    if (!selectedObject) {
        container.innerHTML = `
            <div class="controls-panel">
                <h3>Object Transform</h3>
                <div class="loading" style="padding: 20px; text-align: center; color: #666;">
                    Select an object to view and edit its transform
                </div>
                <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                    <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">Scene Objects</h4>
                    ${getSceneObjectsListHTML()}
                </div>
                <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                    <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">ArUco Markers</h4>
                    ${getArUcoMarkersListHTML()}
                </div>
            </div>
        `;
        return;
    }

    const obj = selectedObject;
    const pos = obj.position;
    const rot = obj.rotation;
    const quat = calculateQuaternion(rot);
    const assemblyType = obj.userData.assemblyType;
    const assemblySubtype = obj.userData.assemblySubtype;
    const pegAxis = obj.userData.pegAxis;

    container.innerHTML = `
        <div class="controls-panel">
            <h3>Object Transform</h3>

            <div class="selected-object" style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 4px;">
                <h4 style="margin: 0 0 5px 0; color: #2c3e50;">${obj.userData.displayName || obj.userData.name}</h4>
                <p style="margin: 0; font-size: 12px; color: #666;">Type: ${obj.userData.type}</p>
            </div>

            <h4 style="margin-top: 15px; margin-bottom: 8px; font-size: 13px; color: #555;">Assembly Type</h4>
            <div class="transform-row" style="margin-bottom: 10px;">
                <select id="assemblyTypeSelect" class="control-input" style="flex: 1;" onchange="setAssemblyType(this.value)">
                    <option value="board" ${assemblyType === 'board' ? 'selected' : ''}>Board</option>
                    <option value="object" ${assemblyType === 'object' ? 'selected' : ''}>Object</option>
                </select>
            </div>
            ${assemblyType === 'object' ? `
            <h4 style="margin-top: 10px; margin-bottom: 8px; font-size: 13px; color: #555;">Subtype</h4>
            <div class="transform-row" style="margin-bottom: 10px;">
                <select id="assemblySubtypeSelect" class="control-input" style="flex: 1;" onchange="setAssemblySubtype(this.value)">
                    <option value="block" ${assemblySubtype === 'block' ? 'selected' : ''}>Block</option>
                    <option value="socket" ${assemblySubtype === 'socket' ? 'selected' : ''}>Socket</option>
                    <option value="peg" ${assemblySubtype === 'peg' ? 'selected' : ''}>Peg</option>
                </select>
            </div>
            ` : ''}
            ${assemblyType === 'object' && assemblySubtype === 'peg' ? `
            <div class="transform-row" style="margin-bottom: 10px;">
                <label>Axis:</label>
                <select id="pegAxisSelect" class="control-input" style="flex: 1;" onchange="setPegAxis(this.value)">
                    <option value="x" ${pegAxis === 'x' ? 'selected' : ''}>X</option>
                    <option value="y" ${pegAxis === 'y' ? 'selected' : ''}>Y</option>
                    <option value="z" ${pegAxis === 'z' ? 'selected' : ''}>Z</option>
                </select>
            </div>
            ` : ''}

            <h4 style="margin-top: 15px; margin-bottom: 8px; font-size: 13px; color: #555;">Position (m)</h4>
            <div class="transform-row">
                <label>X:</label>
                <input type="number" id="objPosX" class="control-input" step="0.0001" value="${pos.x.toFixed(4)}"
                       onchange="setObjectPosition('x', parseFloat(this.value))">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn x" onclick="moveObject('x', -0.01)">-</button>
                    <button class="btn btn-small axis-btn x" onclick="moveObject('x', 0.01)">+</button>
                </div>
            </div>
            <div class="transform-row">
                <label>Y:</label>
                <input type="number" id="objPosY" class="control-input" step="0.0001" value="${pos.y.toFixed(4)}"
                       onchange="setObjectPosition('y', parseFloat(this.value))">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn y" onclick="moveObject('y', -0.01)">-</button>
                    <button class="btn btn-small axis-btn y" onclick="moveObject('y', 0.01)">+</button>
                </div>
            </div>
            <div class="transform-row">
                <label>Z:</label>
                <input type="number" id="objPosZ" class="control-input" step="0.0001" value="${pos.z.toFixed(4)}"
                       onchange="setObjectPosition('z', parseFloat(this.value))">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn z" onclick="moveObject('z', -0.01)">-</button>
                    <button class="btn btn-small axis-btn z" onclick="moveObject('z', 0.01)">+</button>
                </div>
            </div>

            <h4 style="margin-top: 12px; margin-bottom: 8px; font-size: 13px; color: #555;">Rotation (deg)</h4>
            <div class="transform-row">
                <label>Roll:</label>
                <input type="number" id="objRotX" class="control-input" step="1" value="${(rot.x * 180 / Math.PI).toFixed(1)}"
                       onchange="setObjectRotation('x', this.value * Math.PI / 180)">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn x" onclick="rotateObject('x', -Math.PI / 36)">-</button>
                    <button class="btn btn-small axis-btn x" onclick="rotateObject('x', Math.PI / 36)">+</button>
                </div>
            </div>
            <div class="transform-row">
                <label>Pitch:</label>
                <input type="number" id="objRotY" class="control-input" step="1" value="${(rot.y * 180 / Math.PI).toFixed(1)}"
                       onchange="setObjectRotation('y', this.value * Math.PI / 180)">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn y" onclick="rotateObject('y', -Math.PI / 36)">-</button>
                    <button class="btn btn-small axis-btn y" onclick="rotateObject('y', Math.PI / 36)">+</button>
                </div>
            </div>
            <div class="transform-row">
                <label>Yaw:</label>
                <input type="number" id="objRotZ" class="control-input" step="1" value="${(rot.z * 180 / Math.PI).toFixed(1)}"
                       onchange="setObjectRotation('z', this.value * Math.PI / 180)">
                <div class="control-buttons">
                    <button class="btn btn-small axis-btn z" onclick="rotateObject('z', -Math.PI / 36)">-</button>
                    <button class="btn btn-small axis-btn z" onclick="rotateObject('z', Math.PI / 36)">+</button>
                </div>
            </div>

            <button class="btn btn-secondary" onclick="resetObjectPosition(); resetObjectRotation();" style="margin-top: 10px; width: 100%;">Reset to Origin</button>

            <h4 style="margin-top: 15px; margin-bottom: 5px; font-size: 14px; color: #555;">Current Pose</h4>
            <div style="background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 12px; font-family: monospace; margin-bottom: 10px;">
                <div><strong>Position:</strong> <span id="objCurrentPos">(${pos.x.toFixed(4)}, ${pos.y.toFixed(4)}, ${pos.z.toFixed(4)})</span></div>
                <div style="margin-top: 5px;"><strong>RPY:</strong> <span id="objCurrentRPY">(${(rot.x * 180 / Math.PI).toFixed(1)}, ${(rot.y * 180 / Math.PI).toFixed(1)}, ${(rot.z * 180 / Math.PI).toFixed(1)})</span></div>
                <div style="margin-top: 5px;"><strong>Quaternion:</strong> <span id="objCurrentQuat">(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})</span></div>
            </div>

            <div class="control-group" style="margin-top: 15px;">
                <h4 style="font-size: 14px; margin-bottom: 8px;">Quick Actions</h4>
                <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                    <button class="btn btn-small" onclick="resetObjectPosition()">Reset Position</button>
                    <button class="btn btn-small" onclick="resetObjectRotation()">Reset Rotation</button>
                    <button class="btn btn-small btn-secondary" onclick="deleteSelectedObject()" style="flex: 1;">Delete</button>
                </div>
            </div>

            <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">Scene Objects</h4>
                ${getSceneObjectsListHTML()}
            </div>
            <div class="control-group" style="margin-top: 20px; border-top: 1px solid #e0e0e0; padding-top: 15px;">
                <h4 style="font-size: 14px; margin-bottom: 10px; color: #555;">ArUco Markers</h4>
                ${getArUcoMarkersListHTML()}
            </div>
        </div>
    `;
}

function updateSceneObjectsList() {
    updateSelectedObjectInfo();
}

function updateStatusBar() {
    document.getElementById('objectCount').textContent = `Objects: ${sceneObjects.length}`;
    document.getElementById('selectedInfo').textContent =
        selectedObject ? `Selected: ${selectedObject.userData.displayName || selectedObject.userData.name}` : 'Selected: None';
}

// Precision control functions
function setObjectPosition(axis, value) {
    if (!selectedObject) return;
    selectedObject.position[axis] = parseFloat(value);
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Set ${selectedObject.userData.displayName} ${axis.toUpperCase()} position: ${value}`);
}

function setObjectRotation(axis, value) {
    if (!selectedObject) return;
    selectedObject.rotation[axis] = parseFloat(value);
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Set ${selectedObject.userData.displayName} ${axis.toUpperCase()} rotation: ${(value * 180 / Math.PI).toFixed(1)}`);
}

function moveObject(axis, delta) {
    if (!selectedObject) return;
    selectedObject.position[axis] += delta;
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Moved ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${selectedObject.position[axis].toFixed(4)}`);
}
window.moveObject = moveObject;

function rotateObject(axis, delta) {
    if (!selectedObject) return;
    selectedObject.rotation[axis] += delta;
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Rotated ${selectedObject.userData.displayName} ${axis.toUpperCase()}: ${(selectedObject.rotation[axis] * 180 / Math.PI).toFixed(1)}`);
}
window.rotateObject = rotateObject;

function setAssemblyType(value) {
    if (!selectedObject) return;
    selectedObject.userData.assemblyType = value;
    if (value === 'board') {
        selectedObject.userData.assemblySubtype = null;
    } else if (!selectedObject.userData.assemblySubtype) {
        selectedObject.userData.assemblySubtype = 'block';
    }
    updateSelectedObjectInfo();
    updateStatus(`Set ${selectedObject.userData.displayName} type: ${value}`);
}
window.setAssemblyType = setAssemblyType;

function setAssemblySubtype(value) {
    if (!selectedObject) return;
    selectedObject.userData.assemblySubtype = value;
    updateSelectedObjectInfo();
    updateStatus(`Set ${selectedObject.userData.displayName} subtype: ${value}`);
}
window.setAssemblySubtype = setAssemblySubtype;

function setPegAxis(value) {
    if (!selectedObject) return;
    selectedObject.userData.pegAxis = value;
    updateSelectedObjectInfo();
    updateStatus(`Set ${selectedObject.userData.displayName} peg axis: ${value}`);
}
window.setPegAxis = setPegAxis;

function updateCurrentPoseDisplay() {
    if (!selectedObject) return;

    const pos = selectedObject.position;
    const rot = selectedObject.rotation;
    const quat = calculateQuaternion(rot);

    const posEl = document.getElementById('objCurrentPos');
    const rpyEl = document.getElementById('objCurrentRPY');
    const quatEl = document.getElementById('objCurrentQuat');

    if (posEl) posEl.textContent = `(${pos.x.toFixed(4)}, ${pos.y.toFixed(4)}, ${pos.z.toFixed(4)})`;
    if (rpyEl) rpyEl.textContent = `(${(rot.x * 180 / Math.PI).toFixed(1)}, ${(rot.y * 180 / Math.PI).toFixed(1)}, ${(rot.z * 180 / Math.PI).toFixed(1)})`;
    if (quatEl) quatEl.textContent = `(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})`;
}

function resetObjectPosition() {
    if (!selectedObject) return;
    selectedObject.position.set(0, 0, 0);
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Reset position for ${selectedObject.userData.displayName}`);
}
window.resetObjectPosition = resetObjectPosition;

function resetObjectRotation() {
    if (!selectedObject) return;
    selectedObject.rotation.set(0, 0, 0);
    updateSelectedObjectInfo();
    updateCurrentPoseDisplay();
    updateStatus(`Reset rotation for ${selectedObject.userData.displayName}`);
}
window.resetObjectRotation = resetObjectRotation;

function deleteSelectedObject() {
    if (!selectedObject) return;

    const objectName = selectedObject.userData.displayName || selectedObject.userData.name;

    scene.remove(selectedObject);

    const index = sceneObjects.indexOf(selectedObject);
    if (index > -1) {
        sceneObjects.splice(index, 1);
    }

    selectedObject = null;

    updateSceneObjectsList();
    updateSelectedObjectInfo();
    updateStatusBar();
    updateStatus(`Deleted ${objectName}`);
    showMessage(`Deleted ${objectName}`, "info");
}
window.deleteSelectedObject = deleteSelectedObject;

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
    updateSceneObjectsList();
    updateSelectedObjectInfo();
    updateStatusBar();
    updateStatus("Scene cleared");
    showMessage("All components removed from scene", "info");
}
window.clearScene = clearScene;

function toggleGrid() {
    gridVisible = !gridVisible;
    gridHelper.visible = gridVisible;
    axesHelper.visible = gridVisible;
    updateStatus(`Grid ${gridVisible ? 'shown' : 'hidden'}`);
}
window.toggleGrid = toggleGrid;

function resetCamera() {
    camera.position.set(0.5, -0.5, 0.5);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.reset();
    updateStatus("Camera reset to default position");
}
window.resetCamera = resetCamera;

async function exportAssembly() {
    const components = sceneObjects
        .filter(obj => obj.userData.type === 'component')
        .map(obj => {
            const rot = obj.rotation;
            const quat = calculateQuaternion(rot);
            const assemblyType = obj.userData.assemblyType;
            const assemblySubtype = obj.userData.assemblySubtype;
            const result = {
                name: obj.userData.name,
                type: assemblyType,
                position: {
                    x: obj.position.x,
                    y: obj.position.y,
                    z: obj.position.z
                },
                rotation: {
                    rpy: {
                        x: rot.x,
                        y: rot.y,
                        z: rot.z
                    },
                    quaternion: {
                        x: quat.x,
                        y: quat.y,
                        z: quat.z,
                        w: quat.w
                    }
                }
            };
            if (assemblyType === 'object') {
                result.subtype = assemblySubtype;
                if (assemblySubtype === 'peg') {
                    result.axis = obj.userData.pegAxis || 'y';
                }
            }
            return result;
        });

    const assembly = {
        timestamp: new Date().toISOString(),
        components: components
    };

    const blob = new Blob([JSON.stringify(assembly, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `assembly_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showMessage("Assembly exported successfully!", "success");
    updateStatus("Assembly exported to downloads");
}
window.exportAssembly = exportAssembly;

function exportPNG() {
    const gridWasVisible = gridHelper && gridHelper.visible;
    const axesWasVisible = axesHelper && axesHelper.visible;
    if (gridHelper) gridHelper.visible = false;
    if (axesHelper) axesHelper.visible = false;

    const originalBackground = scene.background;
    scene.background = null;

    const colorChanges = [];
    const hiddenMarkers = [];
    sceneObjects.forEach(obj => {
        if (obj.userData) {
            if (obj.userData.type === 'marker') {
                if (obj.visible) {
                    obj.visible = false;
                    hiddenMarkers.push(obj);
                }
            } else if (obj.material) {
                const originalColor = obj.material.color.getHex();
                if (obj.userData.type === 'component') {
                    obj.material.color.setHex(0x000000);
                    colorChanges.push({ obj, originalColor });
                } else if (obj.userData.type === 'grasp_point') {
                    obj.material.color.setHex(0xff0000);
                    if (obj.material.emissive) {
                        obj.material.emissive.setHex(0xff0000);
                    }
                    colorChanges.push({ obj, originalColor, isGrasp: true });
                }
            }
        }
    });

    const exportRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true
    });
    exportRenderer.setSize(1920, 1080);
    exportRenderer.setClearColor(0x000000, 0);
    exportRenderer.shadowMap.enabled = true;
    exportRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const originalLeft = camera.left;
    const originalRight = camera.right;
    const originalTop = camera.top;
    const originalBottom = camera.bottom;
    const frustumSize = 0.5;
    const exportAspect = 1920 / 1080;
    camera.left = frustumSize * exportAspect / -2;
    camera.right = frustumSize * exportAspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = frustumSize / -2;
    camera.updateProjectionMatrix();

    exportRenderer.render(scene, camera);

    const dataURL = exportRenderer.domElement.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = `scene_${new Date().toISOString().split('T')[0]}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    camera.left = originalLeft;
    camera.right = originalRight;
    camera.top = originalTop;
    camera.bottom = originalBottom;
    camera.updateProjectionMatrix();

    scene.background = originalBackground;

    colorChanges.forEach(({ obj, originalColor, isGrasp }) => {
        obj.material.color.setHex(originalColor);
        if (isGrasp && obj.material.emissive) {
            obj.material.emissive.setHex(originalColor);
        }
    });

    hiddenMarkers.forEach(marker => {
        marker.visible = true;
    });

    if (gridHelper) gridHelper.visible = gridWasVisible;
    if (axesHelper) axesHelper.visible = axesWasVisible;

    exportRenderer.dispose();

    showMessage("PNG exported (1920x1080, transparent)", "success");
    updateStatus("PNG image exported to downloads");
}
window.exportPNG = exportPNG;

async function renderArUcoMarkers() {
    const markerObjects = sceneObjects.filter(obj =>
        obj.userData && obj.userData.type === 'marker'
    );

    if (markerObjects.length === 0) {
        showMessage("No ArUco markers in scene to render", "info");
        return;
    }

    updateStatus("Rendering ArUco marker textures...");
    showMessage("Loading ArUco textures...", "info");

    const textureLoader = new THREE.TextureLoader();
    let renderedCount = 0;

    for (const marker of markerObjects) {
        const arucoId = marker.userData.arucoId;
        if (arucoId === undefined) continue;

        const dictionary = 'DICT_4X4_50';
        const markerImageUrl = `/api/marker-image?dictionary=${encodeURIComponent(dictionary)}&marker_id=${arucoId}&size=512`;

        try {
            const texture = await new Promise((resolve, reject) => {
                textureLoader.load(
                    markerImageUrl,
                    (texture) => {
                        texture.wrapS = THREE.ClampToEdgeWrapping;
                        texture.wrapT = THREE.ClampToEdgeWrapping;
                        texture.minFilter = THREE.LinearMipmapLinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        texture.generateMipmaps = true;
                        texture.flipY = true;
                        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
                        resolve(texture);
                    },
                    undefined,
                    (error) => {
                        console.warn('Failed to load marker texture:', error);
                        resolve(null);
                    }
                );
            });

            if (texture) {
                const newMaterial = new THREE.MeshStandardMaterial({
                    map: texture,
                    color: 0xffffff,
                    side: THREE.DoubleSide,
                    transparent: false,
                    depthWrite: true,
                    depthTest: true
                });

                if (marker.material) {
                    marker.material.dispose();
                }
                marker.material = newMaterial;
                marker.userData.originalColor = 0xffffff;
                renderedCount++;
            }
        } catch (error) {
            console.error('Error rendering marker:', error);
        }
    }

    showMessage(`Rendered ${renderedCount} ArUco marker textures`, "success");
    updateStatus(`ArUco textures applied to ${renderedCount} markers`);
}
window.renderArUcoMarkers = renderArUcoMarkers;

function unrenderArUcoMarkers() {
    const markerObjects = sceneObjects.filter(obj =>
        obj.userData && obj.userData.type === 'marker'
    );

    if (markerObjects.length === 0) {
        showMessage("No ArUco markers in scene", "info");
        return;
    }

    let resetCount = 0;
    for (const marker of markerObjects) {
        if (marker.material) {
            if (marker.material.map) {
                marker.material.map.dispose();
            }
            marker.material.dispose();
        }

        marker.material = new THREE.MeshBasicMaterial({
            color: 0xff6b6b,
            transparent: true,
            opacity: 0.7
        });
        marker.userData.originalColor = 0xff6b6b;
        resetCount++;
    }

    showMessage(`Reset ${resetCount} ArUco markers`, "success");
    updateStatus(`ArUco markers reset to default`);
}
window.unrenderArUcoMarkers = unrenderArUcoMarkers;

// Collision Detection Functions
function checkCollisions() {
    const components = sceneObjects.filter(obj =>
        obj.userData && obj.userData.type === 'component'
    );

    if (components.length < 2) {
        showMessage("Need at least 2 components to check collisions", "info");
        return;
    }

    clearCollisionHighlights();

    const collidingPairs = [];
    for (let i = 0; i < components.length; i++) {
        for (let j = i + 1; j < components.length; j++) {
            if (checkObjectsCollide(components[i], components[j])) {
                collidingPairs.push([components[i], components[j]]);
            }
        }
    }

    if (collidingPairs.length === 0) {
        showMessage("No collisions detected!", "success");
        updateStatus("No collisions found");
    } else {
        const collidingObjects = new Set();
        collidingPairs.forEach(pair => {
            collidingObjects.add(pair[0]);
            collidingObjects.add(pair[1]);
        });

        collidingObjects.forEach(obj => {
            highlightCollision(obj);
        });

        const collisionList = collidingPairs.map(pair =>
            `${pair[0].userData.displayName} - ${pair[1].userData.displayName}`
        ).join(', ');

        showMessage(`${collidingPairs.length} collision(s) detected: ${collisionList}`, "error");
        updateStatus(`Found ${collidingPairs.length} collision(s)`);
    }
}
window.checkCollisions = checkCollisions;

function checkObjectsCollide(obj1, obj2) {
    const box1 = new THREE.Box3().setFromObject(obj1);
    const box2 = new THREE.Box3().setFromObject(obj2);

    if (!box1.intersectsBox(box2)) {
        return false;
    }

    const edges1 = getWorldSpaceEdges(obj1);
    const edges2 = getWorldSpaceEdges(obj2);

    for (const edge1 of edges1) {
        for (const edge2 of edges2) {
            if (edgesIntersect(edge1, edge2)) {
                return true;
            }
        }
    }

    return false;
}

function getWorldSpaceEdges(obj) {
    const edges = [];
    const positionAttribute = obj.geometry.getAttribute('position');
    if (!positionAttribute) return edges;

    for (let i = 0; i < positionAttribute.count; i += 2) {
        const v1 = new THREE.Vector3();
        const v2 = new THREE.Vector3();

        v1.fromBufferAttribute(positionAttribute, i);
        v2.fromBufferAttribute(positionAttribute, i + 1);

        v1.applyMatrix4(obj.matrixWorld);
        v2.applyMatrix4(obj.matrixWorld);

        edges.push({ start: v1, end: v2 });
    }

    return edges;
}

function edgesIntersect(edge1, edge2) {
    const p1 = edge1.start;
    const p2 = edge1.end;
    const p3 = edge2.start;
    const p4 = edge2.end;

    const d1 = new THREE.Vector3().subVectors(p2, p1);
    const d2 = new THREE.Vector3().subVectors(p4, p3);
    const r = new THREE.Vector3().subVectors(p1, p3);

    const a = d1.dot(d1);
    const b = d1.dot(d2);
    const c = d2.dot(d2);
    const d = d1.dot(r);
    const e = d2.dot(r);

    const denom = a * c - b * b;
    const touchTolerance = 0.0001;

    let s, t;

    if (Math.abs(denom) < 1e-10) {
        s = 0;
        t = d / b;

        const c1 = new THREE.Vector3().copy(p1).addScaledVector(d1, s);
        const c2 = new THREE.Vector3().copy(p3).addScaledVector(d2, Math.max(0, Math.min(1, t)));
        const distance = c1.distanceTo(c2);

        if (distance < touchTolerance) {
            return checkParallelSegmentOverlap(p1, p2, p3, p4, touchTolerance);
        }
        return false;
    }

    s = (b * e - c * d) / denom;
    t = (a * e - b * d) / denom;

    const sInBounds = s >= -touchTolerance && s <= 1 + touchTolerance;
    const tInBounds = t >= -touchTolerance && t <= 1 + touchTolerance;

    if (!sInBounds || !tInBounds) {
        s = Math.max(0, Math.min(1, s));
        t = Math.max(0, Math.min(1, t));

        if (s <= 0 || s >= 1) {
            t = (s <= 0) ? -d / b : (a - d) / b;
            t = Math.max(0, Math.min(1, t));
        }
        if (t <= 0 || t >= 1) {
            s = (t <= 0) ? -e / a : (b - e) / a;
            s = Math.max(0, Math.min(1, s));
        }
    }

    const c1 = new THREE.Vector3().copy(p1).addScaledVector(d1, s);
    const c2 = new THREE.Vector3().copy(p3).addScaledVector(d2, t);

    return c1.distanceTo(c2) < touchTolerance;
}

function checkParallelSegmentOverlap(p1, p2, p3, p4, tolerance) {
    const dir = new THREE.Vector3().subVectors(p2, p1).normalize();

    const t1 = 0;
    const t2 = new THREE.Vector3().subVectors(p2, p1).dot(dir);
    const t3 = new THREE.Vector3().subVectors(p3, p1).dot(dir);
    const t4 = new THREE.Vector3().subVectors(p4, p1).dot(dir);

    const linePoint = new THREE.Vector3().copy(p1).addScaledVector(dir, t3);
    const distToLine = p3.distanceTo(linePoint);

    if (distToLine > tolerance) {
        return false;
    }

    const min1 = Math.min(t1, t2);
    const max1 = Math.max(t1, t2);
    const min2 = Math.min(t3, t4);
    const max2 = Math.max(t3, t4);

    return Math.max(min1, min2) <= Math.min(max1, max2) + tolerance;
}

function highlightCollision(object) {
    if (object.material) {
        if (!object.userData.collisionOriginalColor) {
            object.userData.collisionOriginalColor = object.material.color.getHex();
        }
        object.material.color.setHex(0xff0000);
        object.userData.isCollisionHighlighted = true;
    }
}

function clearCollisionHighlights() {
    sceneObjects.forEach(obj => {
        if (obj.userData && obj.userData.isCollisionHighlighted) {
            if (obj.material && obj.userData.collisionOriginalColor !== undefined) {
                obj.material.color.setHex(obj.userData.collisionOriginalColor);
                delete obj.userData.collisionOriginalColor;
                delete obj.userData.isCollisionHighlighted;
            }
        }
    });
}

async function loadAssemblyFromFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        updateStatus("Loading assembly from file...");
        showMessage("Reading assembly file...", "info");

        const text = await file.text();
        const assemblyData = JSON.parse(text);

        if (!assemblyData.components || !Array.isArray(assemblyData.components)) {
            throw new Error("Invalid assembly file format - missing components array");
        }

        clearScene();

        if (Object.keys(loadedComponents).length === 0) {
            await loadAllComponents();
        }

        let loadedCount = 0;
        for (const componentData of assemblyData.components) {
            await restoreComponentFromAssembly(componentData);
            loadedCount++;
        }

        updateSceneObjectsList();
        updateStatusBar();
        showMessage(`Successfully loaded assembly with ${loadedCount} components!`, "success");
        updateStatus(`Assembly loaded: ${loadedCount} components restored`);

    } catch (error) {
        showMessage(`Error loading assembly: ${error.message}`, "error");
        updateStatus("Error loading assembly file");
        console.error("Assembly loading error:", error);
    }

    event.target.value = '';
}

async function restoreComponentFromAssembly(componentData) {
    const componentName = componentData.name;
    const component = loadedComponents[componentName];

    if (!component) {
        console.warn(`Component ${componentName} not found in loaded components`);
        return;
    }

    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(component.wireframe.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    const edges = component.wireframe.edges;
    const lineGeometry = new THREE.BufferGeometry();
    const lineVertices = [];

    edges.forEach(edge => {
        const v1 = component.wireframe.vertices[edge[0]];
        const v2 = component.wireframe.vertices[edge[1]];
        lineVertices.push(...v1, ...v2);
    });

    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));

    const material = new THREE.LineBasicMaterial({
        color: getComponentColor(componentName),
        linewidth: 1
    });

    const wireframe = new THREE.LineSegments(lineGeometry, material);
    wireframe.userData = {
        name: componentName,
        type: 'component',
        displayName: component.display_name,
        originalColor: getComponentColor(componentName),
        id: generateId(),
        assemblyType: componentData.type,
        assemblySubtype: componentData.subtype || null,
        pegAxis: componentData.axis || 'y'
    };

    wireframe.position.set(
        componentData.position.x,
        componentData.position.y,
        componentData.position.z
    );

    if (!componentData.rotation || !componentData.rotation.rpy) {
        throw new Error(`Assembly component ${componentName} is missing rotation.rpy (expected format: rotation.rpy.{x,y,z})`);
    }

    const rotX = componentData.rotation.rpy.x;
    const rotY = componentData.rotation.rpy.y;
    const rotZ = componentData.rotation.rpy.z;

    wireframe.rotation.set(rotX, rotY, rotZ);

    scene.add(wireframe);
    sceneObjects.push(wireframe);

    if (component.aruco && component.aruco.markers) {
        component.aruco.markers.forEach((marker, index) => {
            addArUcoMarker(marker, wireframe, index);
        });
    }
}

function showFloatingControls() {
    document.getElementById('floatingControls').style.display = 'block';
    document.getElementById('showControlsBtn').style.display = 'none';
}
window.showFloatingControls = showFloatingControls;

function hideFloatingControls() {
    document.getElementById('floatingControls').style.display = 'none';
    document.getElementById('showControlsBtn').style.display = 'block';
}
window.hideFloatingControls = hideFloatingControls;

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

    setTimeout(() => {
        if (div.parentNode) {
            div.parentNode.removeChild(div);
        }
    }, 4000);
}

// Grasp Points Visualization Functions
async function loadGraspPointsFromData(tempData, targetObject, clearFirst = false) {
    if (!targetObject || !targetObject.userData || targetObject.userData.type !== 'component') {
        throw new Error("No valid target object selected for grasp points (expected a component).");
    }

    if (!tempData.markers || !Array.isArray(tempData.markers)) {
        throw new Error("Invalid grasp points data - missing markers array");
    }

    if (!tempData.wireframe || !tempData.wireframe.vertices || !tempData.wireframe.edges) {
        throw new Error("Invalid grasp points data - missing wireframe data");
    }

    if (!tempData.grasp_points || !Array.isArray(tempData.grasp_points)) {
        throw new Error("Invalid grasp points data - missing grasp_points array");
    }

    if (clearFirst) {
        clearGraspPoints();
    }

    graspPointsData = tempData;

    let totalPoints = 0;
    graspPointsData.grasp_points.forEach((graspPoint, idx) => {
        const sphere = createGraspPointSphere(graspPoint, idx);
        targetObject.add(sphere);
        sceneObjects.push(sphere);
        totalPoints++;
    });

    const objName = targetObject.userData.displayName || targetObject.userData.name;
    const infoText = `Loaded ${objName}: ${totalPoints} grasp points`;
    document.getElementById('graspInfo').innerHTML = `
        <span style="color: #27ae60; font-weight: 500;">${infoText}</span>
    `;

    showMessage(infoText, "success");
    updateStatus(`Grasp visualization loaded for ${objName}`);
}

async function loadGraspPointsFromFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        if (!selectedObject || !selectedObject.userData || selectedObject.userData.type !== 'component') {
            showMessage("Please select a component in the scene before loading grasp points from file.", "info");
            return;
        }

        updateStatus("Loading grasp points from file...");
        showMessage("Reading grasp points file...", "info");

        const text = await file.text();
        const tempData = JSON.parse(text);

        await loadGraspPointsFromData(tempData, selectedObject, true);
    } catch (error) {
        showMessage(`Error loading grasp points: ${error.message}`, "error");
        updateStatus("Error loading grasp points");
        console.error("Grasp points loading error:", error);
    }

    event.target.value = '';
}

async function loadGraspPointsAuto() {
    try {
        const componentObjects = sceneObjects.filter(obj =>
            obj.userData && obj.userData.type === 'component'
        );

        if (componentObjects.length === 0) {
            showMessage("No components in scene - load an assembly or add components first.", "info");
            updateStatus("No components in scene - cannot load grasp points.");
            return;
        }

        updateStatus("Loading grasp points for all scene components from data directory...");
        showMessage("Loading grasp points for all scene components from server...", "info");

        clearGraspPoints();

        let anyLoaded = false;

        for (const obj of componentObjects) {
            const objectName = obj.userData.name;
            try {
                const response = await fetch(`/api/grasp-points/${objectName}`);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.warn(`Grasp data not found for ${objectName}: ${errorText}`);
                    continue;
                }

                const data = await response.json();
                await loadGraspPointsFromData(data, obj, false);
                anyLoaded = true;
            } catch (innerError) {
                console.warn(`Grasp points auto-load error for ${objectName}:`, innerError);
            }
        }

        if (!anyLoaded) {
            showMessage("No grasp data found in data/grasp for any scene components.", "info");
            updateStatus("No grasp points loaded.");
        } else {
            updateStatus("Grasp points auto-loaded for available components.");
        }
    } catch (error) {
        console.error("Grasp points auto-load error:", error);
        showMessage(`Error auto-loading grasp points: ${error.message}`, "error");
        updateStatus("Error auto-loading grasp points.");
    }
}

function createGraspPointSphere(graspPoint, index) {
    const geometry = new THREE.SphereGeometry(0.003, 16, 16);
    const material = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        emissive: 0x00ff00,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.9
    });

    const sphere = new THREE.Mesh(geometry, material);

    const pos = graspPoint.position;
    sphere.position.set(pos.x, pos.y, pos.z);

    sphere.userData = {
        name: `Grasp-${graspPoint.id}`,
        type: 'grasp_point',
        graspId: graspPoint.id,
        displayName: `Grasp Point ${graspPoint.id}`,
        originalColor: 0x00ff00,
        id: generateId()
    };

    return sphere;
}

function clearGraspPoints() {
    const toRemove = sceneObjects.filter(obj =>
        obj.userData &&
        (
            obj.userData.type === 'grasp_wireframe' ||
            obj.userData.type === 'grasp_marker' ||
            obj.userData.type === 'grasp_point'
        )
    );

    toRemove.forEach(obj => {
        if (obj.parent) {
            obj.parent.remove(obj);
        }
        const index = sceneObjects.indexOf(obj);
        if (index > -1) {
            sceneObjects.splice(index, 1);
        }
    });

    graspPointsGroup = null;
    graspPointsData = null;

    if (selectedObject && (
        selectedObject.userData.type === 'grasp_wireframe' ||
        selectedObject.userData.type === 'grasp_marker' ||
        selectedObject.userData.type === 'grasp_point'
    )) {
        selectedObject = null;
    }

    updateSceneObjectsList();
    document.getElementById('graspInfo').innerHTML = 'No grasp points loaded';
    updateStatus("Grasp points cleared");
    showMessage("Grasp points visualization cleared", "info");
}
window.clearGraspPoints = clearGraspPoints;
window.loadAllComponents = loadAllComponents;

// Initialize when page loads
window.addEventListener('load', initScene);
