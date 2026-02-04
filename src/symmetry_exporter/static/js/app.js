/**
 * Symmetry Exporter - 3D Visualization App
 * Interactive 3D environment for displaying individual objects with grasp points
 */

// Global variables
let scene, camera, renderer, controls;
let loadedComponents = {};
let sceneObjects = [];
let selectedObject = null;
let gridVisible = true;
let gridHelper;
let objectAxesHelper = null;
let raycaster, mouse;
let isMouseDown = false;
let graspPointsGroup = null;
let graspPointsData = null;
let currentObjectName = null;
let graspPointsList = [];
let currentWireframe = null;
let groundTruthWireframe = null;
let objectFoldValues = {};
const SYMMETRY_COLOR = 0x00ff00;
let canonicalAnchor = { active: false, axes: [], euler: { x: 0, y: 0, z: 0 } };

// Grasp point colors
const GRASP_COLOR_DEFAULT = 0x00ff00;
const GRASP_COLOR_SELECTED = 0xff0000;
const GRIPPER_COLOR_DEFAULT = 0xffaa00;
const GRIPPER_COLOR_SELECTED = 0xff0000;

// Initialize the 3D scene
function initScene() {
    const container = document.getElementById('viewer');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

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

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minZoom = 0.1;
    controls.maxZoom = 50;

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

    gridHelper = new THREE.GridHelper(1, 1, 0x444444, 0x222222);
    gridHelper.rotateX(Math.PI / 2);
    scene.add(gridHelper);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    setupEventListeners();
    animate();
    updateStatus("Scene initialized - Ready to load components");
}

function setupEventListeners() {
    const canvas = renderer.domElement;
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('resize', onWindowResize);
    document.addEventListener('keydown', onKeyDown);
}

function onMouseDown(event) {
    isMouseDown = true;
}

function onMouseUp(event) {
    if (!isMouseDown) return;
    isMouseDown = false;

    if (event.button === 0) {
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

    const rotStep = event.shiftKey ? Math.PI / 18 : Math.PI / 36;

    switch(event.key) {
        case 'q':
        case 'Q':
            if (selectedObject.userData && selectedObject.userData.type === 'component') {
                selectedObject.rotateY(-rotStep);
                const ang = selectedObject.userData.sliderAngles || { x:0, y:0, z:0 };
                ang.y = (ang.y || 0) - rotStep;
                selectedObject.userData.sliderAngles = ang;
            }
            break;
        case 'e':
        case 'E':
            if (selectedObject.userData && selectedObject.userData.type === 'component') {
                selectedObject.rotateY(rotStep);
                const ang = selectedObject.userData.sliderAngles || { x:0, y:0, z:0 };
                ang.y = (ang.y || 0) + rotStep;
                selectedObject.userData.sliderAngles = ang;
            }
            break;
        case 'Delete':
            deleteSelectedObject();
            break;
    }

    if (['q', 'e', 'Q', 'E'].includes(event.key)) {
        event.preventDefault();
        checkFoldSymmetry('y');
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

    if (sceneObjects.length > 0 || currentObjectName !== null) {
        clearScene();
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
        linewidth: 2
    });

    const wireframe = new THREE.LineSegments(lineGeometry, material);
    wireframe.userData = {
        name: componentName,
        type: 'component',
        displayName: component.display_name,
        originalColor: getComponentColor(componentName),
        id: generateId(),
        sliderAngles: { x: 0, y: 0, z: 0 }
    };

    wireframe.position.set(0, 0, 0);

    scene.add(wireframe);
    sceneObjects.push(wireframe);

    currentWireframe = wireframe;

    // Create ground truth wireframe
    const groundTruthMaterial = new THREE.LineBasicMaterial({
        color: getComponentColor(componentName),
        linewidth: 2,
        opacity: 0.2,
        transparent: true
    });

    const groundTruthGeometry = lineGeometry.clone();
    const groundTruthWireframeObj = new THREE.LineSegments(groundTruthGeometry, groundTruthMaterial);
    groundTruthWireframeObj.userData = {
        name: componentName + '_groundtruth',
        type: 'groundtruth',
        displayName: component.display_name + ' (Ground Truth)',
        id: generateId()
    };

    groundTruthWireframeObj.position.set(0, 0, 0);
    groundTruthWireframeObj.rotation.set(0, 0, 0);
    groundTruthWireframeObj.renderOrder = -1;

    scene.add(groundTruthWireframeObj);
    groundTruthWireframe = groundTruthWireframeObj;

    if (component.aruco && component.aruco.markers) {
        component.aruco.markers.forEach((marker, index) => {
            addArUcoMarker(marker, wireframe, index);
        });
    }

    currentObjectName = componentName;
    loadGraspPointsForObject(componentName);

    updateStatus(`Added ${component.display_name} to scene`);
    showMessage(`Added ${component.display_name} to scene`, "success");

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
    if (objectAxesHelper && objectAxesHelper.parent) {
        objectAxesHelper.parent.remove(objectAxesHelper);
        objectAxesHelper = null;
    }

    if (selectedObject && selectedObject.material) {
        if (selectedObject.userData.type === 'grasp_point') {
            selectedObject.material.color.setHex(GRASP_COLOR_DEFAULT);
            selectedObject.material.emissive.setHex(GRASP_COLOR_DEFAULT);
        } else {
            selectedObject.material.color.setHex(selectedObject.userData.originalColor);
        }
    }

    selectedObject = object;

    if (selectedObject) {
        if (selectedObject.material) {
            if (selectedObject.userData.type === 'grasp_point') {
                selectedObject.material.color.setHex(GRASP_COLOR_SELECTED);
                selectedObject.material.emissive.setHex(GRASP_COLOR_SELECTED);
            } else {
                selectedObject.material.color.setHex(0xffff00);
            }
        }

        if (selectedObject.userData.type === 'component') {
            objectAxesHelper = new THREE.AxesHelper(0.1);
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

function normalizeAngleForSlider(angleRad) {
    let angleDeg = (angleRad * 180 / Math.PI);
    angleDeg = angleDeg % 360;
    if (angleDeg < 0) angleDeg += 360;
    return Math.round(angleDeg / 5) * 5;
}

function getQuaternionDisplayFromObject(obj) {
    const q = obj.quaternion;
    return `X: ${q.x.toFixed(4)}, Y: ${q.y.toFixed(4)}, Z: ${q.z.toFixed(4)}, W: ${q.w.toFixed(4)}`;
}

function updateSelectedObjectInfo() {
    const container = document.getElementById('selectedObjectInfo');
    const foldPanel = document.getElementById('foldSymmetryPanel');

    if (!selectedObject) {
        container.innerHTML = '';
        if (foldPanel) foldPanel.style.display = 'none';
        return;
    }

    const obj = selectedObject;
    const angles = obj.userData && obj.userData.sliderAngles ? obj.userData.sliderAngles : obj.rotation;

    if (obj.userData.type === 'component' && foldPanel) {
        foldPanel.style.display = 'block';
        restoreFoldValues();
    } else if (foldPanel) {
        foldPanel.style.display = 'none';
    }

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
                    <input type="range" class="control-input" min="0" max="360" step="5"
                           value="${normalizeAngleForSlider(angles.x)}"
                           oninput="setObjectRotationFromSlider('x', this.value)"
                           style="flex: 1; margin: 0 10px;">
                    <input type="number" class="control-input" step="5" value="${Math.round((angles.x * 180 / Math.PI) / 5) * 5}"
                           onchange="setObjectRotation('x', this.value * Math.PI / 180)"
                           style="width: 80px;">
                </div>
                <div class="control-row">
                    <span class="control-label">Y:</span>
                    <input type="range" class="control-input" min="0" max="360" step="5"
                           value="${normalizeAngleForSlider(angles.y)}"
                           oninput="setObjectRotationFromSlider('y', this.value)"
                           style="flex: 1; margin: 0 10px;">
                    <input type="number" class="control-input" step="5" value="${Math.round((angles.y * 180 / Math.PI) / 5) * 5}"
                           onchange="setObjectRotation('y', this.value * Math.PI / 180)"
                           style="width: 80px;">
                </div>
                <div class="control-row">
                    <span class="control-label">Z:</span>
                    <input type="range" class="control-input" min="0" max="360" step="5"
                           value="${normalizeAngleForSlider(angles.z)}"
                           oninput="setObjectRotationFromSlider('z', this.value)"
                           style="flex: 1; margin: 0 10px;">
                    <input type="number" class="control-input" step="5" value="${Math.round((angles.z * 180 / Math.PI) / 5) * 5}"
                           onchange="setObjectRotation('z', this.value * Math.PI / 180)"
                           style="width: 80px;">
                </div>
            </div>

            <div class="control-group">
                <h4>Quaternion</h4>
                <div id="quaternionDisplay" style="padding: 10px; background: #f8f9fa; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 13px;">
                    ${getQuaternionDisplayFromObject(obj)}
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

function updateSceneObjectsList() {
    // Compatibility function
}

function setObjectRotation(axis, value) {
    if (!selectedObject) return;
    if (selectedObject.userData.type !== 'component') return;
    const obj = selectedObject;
    const degrees = Math.round(parseFloat(value) * 180 / Math.PI / 5) * 5;
    const radians = degrees * Math.PI / 180;
    const angles = obj.userData.sliderAngles || { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z };
    const prev = angles[axis] || 0;
    const delta = radians - prev;
    if (axis === 'x') obj.rotateX(delta);
    if (axis === 'y') obj.rotateY(delta);
    if (axis === 'z') obj.rotateZ(delta);
    angles[axis] = radians;
    obj.userData.sliderAngles = angles;
    checkFoldSymmetry(axis);
    updateSelectedObjectInfo();
    updateStatus(`Rotated ${obj.userData.displayName} ${axis.toUpperCase()}: ${degrees}`);
}

function setObjectRotationFromSlider(axis, value) {
    if (!selectedObject) return;
    if (selectedObject.userData.type !== 'component') return;
    const obj = selectedObject;
    const degrees = Math.round(parseFloat(value) / 5) * 5;
    const radians = degrees * Math.PI / 180;
    const angles = obj.userData.sliderAngles || { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z };
    const prev = angles[axis] || 0;
    const delta = radians - prev;
    if (axis === 'x') obj.rotateX(delta);
    if (axis === 'y') obj.rotateY(delta);
    if (axis === 'z') obj.rotateZ(delta);
    angles[axis] = radians;
    obj.userData.sliderAngles = angles;
    checkFoldSymmetry(axis);

    const infoContainer = document.getElementById('selectedObjectInfo');
    if (infoContainer) {
        const numberInput = infoContainer.querySelector(`input[type="number"][onchange*="${axis}"]`);
        if (numberInput) {
            numberInput.value = degrees;
        }
    }

    const quaternionDisplay = document.getElementById('quaternionDisplay');
    if (quaternionDisplay) {
        quaternionDisplay.innerHTML = getQuaternionDisplayFromObject(obj);
    }
}

function rotateObject(axis, delta) {
    if (!selectedObject) return;
    if (selectedObject.userData.type !== 'component') return;
    const obj = selectedObject;
    if (axis === 'x') obj.rotateX(delta);
    if (axis === 'y') obj.rotateY(delta);
    if (axis === 'z') obj.rotateZ(delta);
    const angles = obj.userData.sliderAngles || { x: 0, y: 0, z: 0 };
    angles[axis] = (angles[axis] || 0) + delta;
    obj.userData.sliderAngles = angles;
    checkFoldSymmetry(axis);
    updateSelectedObjectInfo();
    updateStatus(`Rotated ${obj.userData.displayName} ${axis.toUpperCase()}: ${(angles[axis] * 180 / Math.PI).toFixed(1)}`);
}

function checkFoldSymmetry(rotatedAxis = null) {
    if (!selectedObject || selectedObject.userData.type !== 'component') return;
    if (!currentObjectName) return;

    const foldValues = objectFoldValues[currentObjectName] || {};
    const rot = selectedObject.userData && selectedObject.userData.sliderAngles ? selectedObject.userData.sliderAngles : selectedObject.rotation;

    const canonicalAxes = [];
    ['x', 'y', 'z'].forEach(axis => {
        const fold = foldValues[axis];
        if (fold && fold >= 2) {
            let angleDeg = (rot[axis] * 180 / Math.PI);
            angleDeg = angleDeg % 360;
            if (angleDeg < 0) angleDeg += 360;
            const stepAngle = 360 / fold;
            if (checkIfSymmetryAngle(angleDeg, fold, stepAngle)) {
                canonicalAxes.push(axis);
            }
        }
    });

    const axesToCheck = rotatedAxis ? [rotatedAxis] : ['x', 'y', 'z'];
    let hasSymmetryForColor = false;
    axesToCheck.forEach(axis => {
        if (canonicalAxes.includes(axis)) {
            hasSymmetryForColor = true;
        }
    });

    if (hasSymmetryForColor && selectedObject.material) {
        selectedObject.material.color.setHex(SYMMETRY_COLOR);
    } else if (selectedObject.material) {
        selectedObject.material.color.setHex(selectedObject.userData.originalColor);
    }

    if (canonicalAxes.length > 0) {
        canonicalAnchor.active = true;
        canonicalAnchor.axes = canonicalAxes.slice();
        canonicalAnchor.euler = { x: rot.x, y: rot.y, z: rot.z };
        restoreGraspPointOrientations();
    } else if (canonicalAnchor.active && canonicalAnchor.axes.length > 0) {
        updateGraspPointOrientationsFromAnchor(canonicalAnchor);
    } else {
        restoreGraspPointOrientations();
    }
}

function restoreGraspPointOrientations() {
    if (!graspPointsList) return;

    graspPointsList.forEach(sphere => {
        if (sphere.userData.gripperGroup && sphere.userData.originalApproachVector) {
            const originalApproach = sphere.userData.originalApproachVector.clone();

            const currentObjectQuat = currentWireframe && currentWireframe.quaternion
                ? currentWireframe.quaternion.clone()
                : new THREE.Quaternion(0, 0, 0, 1);
            const inverseQuat = currentObjectQuat.clone().invert();
            const rotatedApproach = originalApproach.applyQuaternion(inverseQuat).normalize();

            const defaultDir = new THREE.Vector3(0, 0, -1);
            const quaternion = new THREE.Quaternion();
            quaternion.setFromUnitVectors(defaultDir, rotatedApproach);
            sphere.userData.gripperGroup.quaternion.copy(quaternion);

            const offsetDistance = 0.008;
            const offsetPosition = rotatedApproach.clone().multiplyScalar(offsetDistance);
            sphere.userData.gripperGroup.position.copy(offsetPosition);
        }
    });
}

function updateGraspPointOrientationsFromAnchor(anchor) {
    if (!currentWireframe || !graspPointsList) return;

    const rotAngles = currentWireframe.userData && currentWireframe.userData.sliderAngles
        ? currentWireframe.userData.sliderAngles
        : currentWireframe.rotation;

    const dx = anchor.axes.includes('x') ? 0 : (rotAngles.x - anchor.euler.x);
    const dy = anchor.axes.includes('y') ? 0 : (rotAngles.y - anchor.euler.y);
    const dz = anchor.axes.includes('z') ? 0 : (rotAngles.z - anchor.euler.z);

    const qTotal = currentWireframe.quaternion.clone();
    const qTotalInv = qTotal.clone().invert();

    const qAnchor = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        anchor.euler.x, anchor.euler.y, anchor.euler.z, 'XYZ'
    ));

    const qResidual = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        dx, dy, dz, 'XYZ'
    ));

    graspPointsList.forEach(sphere => {
        if (sphere.userData.gripperGroup && sphere.userData.originalApproachVector) {
            const originalApproach = sphere.userData.originalApproachVector.clone();
            const defaultDir = new THREE.Vector3(0, 0, -1);
            const qOriginal = new THREE.Quaternion().setFromUnitVectors(defaultDir, originalApproach);
            const qWorldCanonical = qAnchor.clone().multiply(qOriginal);

            const desiredWorld = qResidual.clone().multiply(qWorldCanonical);
            const localQuat = qTotalInv.clone().multiply(desiredWorld);
            sphere.userData.gripperGroup.quaternion.copy(localQuat);

            const offsetDistance = 0.008;
            const canonicalApproachWorld = originalApproach.clone().applyQuaternion(qAnchor).normalize();
            const canonicalApproachLocal = canonicalApproachWorld.clone().applyQuaternion(qTotalInv).normalize();
            const offsetPosition = canonicalApproachLocal.multiplyScalar(offsetDistance);
            sphere.userData.gripperGroup.position.copy(offsetPosition);
        }
    });
}

function checkIfSymmetryAngle(angleDeg, fold, stepAngle) {
    const tolerance = 1.0;
    for (let i = 0; i < fold; i++) {
        const symmetryAngle = i * stepAngle;
        const diff = Math.abs(angleDeg - symmetryAngle);
        const diffWrapped = Math.min(diff, 360 - diff);
        if (diffWrapped <= tolerance) {
            return true;
        }
    }
    return false;
}

function updateFoldValue(axis, value) {
    if (!currentObjectName) return;

    const foldNum = parseInt(value) || 0;

    if (!objectFoldValues[currentObjectName]) {
        objectFoldValues[currentObjectName] = {};
    }

    if (foldNum > 0) {
        objectFoldValues[currentObjectName][axis] = foldNum;
    } else {
        delete objectFoldValues[currentObjectName][axis];
    }

    checkFoldSymmetry();
    updateStatus(`Set ${axis.toUpperCase()}-axis fold: ${foldNum > 0 ? foldNum : 'none'}`);
}

function resetObjectRotation() {
    if (!selectedObject) return;
    selectedObject.quaternion.set(0, 0, 0, 1);
    if (selectedObject.userData) {
        selectedObject.userData.sliderAngles = { x: 0, y: 0, z: 0 };
    }
    checkFoldSymmetry();
    updateSelectedObjectInfo();
    updateStatus(`Reset rotation for ${selectedObject.userData.displayName}`);
}

function restoreFoldValues() {
    if (!currentObjectName) return;

    const foldValues = objectFoldValues[currentObjectName] || {};
    document.getElementById('foldXInput').value = foldValues.x || '';
    document.getElementById('foldYInput').value = foldValues.y || '';
    document.getElementById('foldZInput').value = foldValues.z || '';
}

async function loadSymmetry(silent = false) {
    if (!currentObjectName) {
        if (!silent) showMessage("No object loaded", "error");
        return;
    }

    try {
        updateStatus("Loading symmetry data...");
        const response = await fetch(`/api/symmetry/${currentObjectName}`);

        if (!response.ok) {
            if (response.status === 404) {
                if (!silent) {
                    showMessage("No symmetry data found for this object", "info");
                }
                updateStatus("No symmetry data available");
                return;
            }
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Load failed');
        }

        const symmetryData = await response.json();
        const foldAxes = symmetryData.fold_axes || {};

        if (!objectFoldValues[currentObjectName]) {
            objectFoldValues[currentObjectName] = {};
        }

        const parsed = {};
        ['x','y','z'].forEach(axis => {
            const val = foldAxes[axis];
            if (val && typeof val === 'object' && typeof val.fold === 'number') {
                parsed[axis] = val.fold;
            } else if (typeof val === 'number') {
                parsed[axis] = val;
            }
        });
        objectFoldValues[currentObjectName] = parsed;

        document.getElementById('foldXInput').value = parsed.x || '';
        document.getElementById('foldYInput').value = parsed.y || '';
        document.getElementById('foldZInput').value = parsed.z || '';

        checkFoldSymmetry();

        if (!silent) {
            const axesCount = Object.keys(foldAxes).length;
            showMessage(`Loaded symmetry data: ${axesCount} axis/axes configured`, "success");
        }
        updateStatus(`Symmetry data loaded - ${Object.keys(foldAxes).length} axis/axes configured`);
    } catch (error) {
        if (!silent) {
            showMessage(`Error loading symmetry: ${error.message}`, "error");
        }
        updateStatus("Load failed");
    }
}

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

function clearScene() {
    sceneObjects.forEach(obj => {
        scene.remove(obj);
    });

    if (groundTruthWireframe) {
        scene.remove(groundTruthWireframe);
        groundTruthWireframe = null;
    }
    canonicalAnchor = { active: false, axes: [], euler: { x: 0, y: 0, z: 0 } };

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
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.reset();
    updateStatus("Camera reset to default position");
}

function computeAxisFoldQuaternions(axis, fold) {
    const results = [];
    for (let i = 0; i < fold; i++) {
        const angleDeg = (360 / fold) * i;
        const angleRad = angleDeg * Math.PI / 180;
        const euler = new THREE.Euler(
            axis === 'x' ? angleRad : 0,
            axis === 'y' ? angleRad : 0,
            axis === 'z' ? angleRad : 0,
            'XYZ'
        );
        const q = new THREE.Quaternion().setFromEuler(euler);
        results.push({
            angle_deg: angleDeg,
            quaternion: {
                x: Number(q.x.toFixed(6)),
                y: Number(q.y.toFixed(6)),
                z: Number(q.z.toFixed(6)),
                w: Number(q.w.toFixed(6)),
            }
        });
    }
    return results;
}

async function exportSymmetry() {
    if (!currentObjectName) {
        showMessage("No object loaded", "error");
        return;
    }

    const foldValues = objectFoldValues[currentObjectName] || {};
    const hasAnyFold = Object.keys(foldValues).length > 0;

    if (!hasAnyFold) {
        showMessage("Please set fold values for at least one axis first", "error");
        return;
    }

    const foldAxesDetailed = {};
    ['x','y','z'].forEach(axis => {
        const f = foldValues[axis];
        if (f && f > 0) {
            foldAxesDetailed[axis] = {
                fold: f,
                quaternions: computeAxisFoldQuaternions(axis, f)
            };
        }
    });

    const exportData = {
        object_name: currentObjectName,
        fold_axes: foldAxesDetailed
    };

    try {
        updateStatus("Exporting symmetry data...");
        const response = await fetch('/api/export-symmetry', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(exportData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Export failed');
        }

        const result = await response.json();
        showMessage(`Symmetry data exported successfully to ${result.filename}`, "success");
        updateStatus(`Export symmetry - ${Object.keys(foldValues).length} axis/axes configured`);
    } catch (error) {
        showMessage(`Error exporting symmetry: ${error.message}`, "error");
        updateStatus("Export failed");
    }
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

    setTimeout(() => {
        if (div.parentNode) {
            div.parentNode.removeChild(div);
        }
    }, 4000);
}

async function loadGraspPointsForObject(objectName) {
    try {
        updateStatus(`Loading grasp points for ${objectName}...`);

        const response = await fetch(`/api/grasp-data/${objectName}`);
        if (!response.ok) {
            if (response.status === 404) {
                updateStatus(`No grasp points file found for ${objectName}`);
                return;
            }
            throw new Error(`Failed to load grasp points: ${response.statusText}`);
        }

        const tempData = await response.json();

        if (!tempData.grasp_points || !Array.isArray(tempData.grasp_points)) {
            throw new Error("Invalid grasp points data - missing grasp_points array");
        }

        clearGraspPoints();

        graspPointsData = tempData;

        graspPointsGroup = new THREE.Group();
        graspPointsGroup.name = "GraspVisualization";

        graspPointsList = [];
        let totalPoints = 0;
        graspPointsData.grasp_points.forEach((graspPoint, idx) => {
            const sphere = createGraspPointSphere(graspPoint, idx);
            graspPointsGroup.add(sphere);
            graspPointsList.push(sphere);
            totalPoints++;
        });

        if (currentWireframe) {
            currentWireframe.add(graspPointsGroup);
        } else {
            scene.add(graspPointsGroup);
        }

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
    const geometry = new THREE.SphereGeometry(0.003, 16, 16);

    const material = new THREE.MeshPhongMaterial({
        color: GRASP_COLOR_DEFAULT,
        emissive: GRASP_COLOR_DEFAULT,
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
        originalColor: GRASP_COLOR_DEFAULT,
        id: generateId(),
        isClickable: true,
        originalApproachVector: graspPoint.approach_vector ?
            new THREE.Vector3(graspPoint.approach_vector.x,
                             graspPoint.approach_vector.y,
                             graspPoint.approach_vector.z).normalize() : null
    };

    if (graspPoint.approach_vector) {
        const gripperGroup = new THREE.Group();

        const gripperWidth = 0.008;
        const gripperLength = 0.012;
        const gripperDepth = 0.004;
        const gripperThickness = 0.001;
        const approachLength = 0.015;

        const gripperMaterial = new THREE.MeshPhongMaterial({
            color: GRIPPER_COLOR_DEFAULT,
            transparent: true,
            opacity: 0.8
        });

        const jawGeometry = new THREE.BoxGeometry(gripperThickness, gripperDepth, gripperLength);

        const leftJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
        leftJaw.position.set(-gripperWidth / 2, 0, 0);
        gripperGroup.add(leftJaw);

        const rightJaw = new THREE.Mesh(jawGeometry, gripperMaterial);
        rightJaw.position.set(gripperWidth / 2, 0, 0);
        gripperGroup.add(rightJaw);

        const palmGeometry = new THREE.BoxGeometry(gripperWidth, gripperDepth * 0.6, gripperThickness);
        const palm = new THREE.Mesh(palmGeometry, gripperMaterial);
        palm.position.set(0, 0, -gripperLength / 2);
        gripperGroup.add(palm);

        const approachGeometry = new THREE.CylinderGeometry(0.0008, 0.0008, approachLength, 8);
        const approachMaterial = new THREE.MeshPhongMaterial({
            color: 0xffaa00,
            transparent: true,
            opacity: 0.8
        });
        const approachLine = new THREE.Mesh(approachGeometry, approachMaterial);
        approachLine.rotation.x = Math.PI / 2;
        approachLine.position.set(0, 0, -gripperLength / 2 - approachLength / 2);
        gripperGroup.add(approachLine);

        const approach = graspPoint.approach_vector;
        const approachVec = new THREE.Vector3(approach.x, approach.y, approach.z).normalize();

        const defaultDir = new THREE.Vector3(0, 0, -1);
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(defaultDir, approachVec);
        gripperGroup.quaternion.copy(quaternion);

        const offsetDistance = 0.008;
        const offsetPosition = approachVec.clone().multiplyScalar(offsetDistance);
        gripperGroup.position.copy(offsetPosition);
        sphere.add(gripperGroup);

        sphere.userData.gripperGroup = gripperGroup;
        sphere.userData.gripperMaterial = gripperMaterial;
    }

    sceneObjects.push(sphere);

    return sphere;
}

function clearGraspPoints() {
    if (graspPointsGroup) {
        graspPointsList.forEach(graspPointObj => {
            const index = sceneObjects.indexOf(graspPointObj);
            if (index > -1) {
                sceneObjects.splice(index, 1);
            }
        });

        if (graspPointsGroup.parent) {
            graspPointsGroup.parent.remove(graspPointsGroup);
        } else {
            scene.remove(graspPointsGroup);
        }
        graspPointsGroup = null;
    }
    graspPointsData = null;
    graspPointsList = [];

    if (selectedObject && selectedObject.userData.type === 'grasp_point') {
        selectedObject = null;
    }

    updateStatusBar();
}

// Attach functions to window for HTML onclick handlers
window.loadAllComponents = loadAllComponents;
window.clearScene = clearScene;
window.loadSymmetry = loadSymmetry;
window.exportSymmetry = exportSymmetry;
window.toggleGrid = toggleGrid;
window.resetCamera = resetCamera;
window.hideFloatingControls = hideFloatingControls;
window.showFloatingControls = showFloatingControls;
window.resetObjectRotation = resetObjectRotation;
window.deleteSelectedObject = deleteSelectedObject;
window.setObjectRotationFromSlider = setObjectRotationFromSlider;
window.setObjectRotation = setObjectRotation;
window.updateFoldValue = updateFoldValue;

// Initialize when page loads
window.addEventListener('load', initScene);
