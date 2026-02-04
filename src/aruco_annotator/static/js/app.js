/**
 * ArUco Grasp Annotator - Main Application JavaScript
 */

// Three.js globals
let scene, camera, renderer, controls;
let meshObject = null;
let markers = [];
let selectedMarkerId = null;
let selectedMarkerMesh = null;
let placementMode = null;
let raycaster, mouse;
let gridHelper, axesHelper;

// Session state
let session_state = {
    current_file: null
};

// Swap marker state
let swapMarker1Id = null;
let swapMarker2Id = null;

/**
 * Initialize the Three.js scene
 */
function init() {
    const container = document.getElementById('viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Orthographic camera for consistent dimensions
    const frustumSize = 0.5;
    const aspect = width / height;
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
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minZoom = 0.1;
    controls.maxZoom = 50;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, -10, 10);
    scene.add(directionalLight);

    // Raycaster for picking
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('click', onMouseClick);

    // Grid (horizontal in XY plane, Z-up)
    gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    gridHelper.rotateX(Math.PI / 2);
    scene.add(gridHelper);

    // Custom axes (Z-up convention)
    const axesLength = 0.5;
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

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onMouseClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    let markerClicked = false;
    if (markers.length > 0) {
        const markerIntersects = raycaster.intersectObjects(markers, false);
        if (markerIntersects.length > 0) {
            const clickedMarker = markerIntersects[0].object;
            const clickedInternalIdRaw = clickedMarker.userData.internalId;
            const clickedInternalId = typeof clickedInternalIdRaw === 'number'
                ? clickedInternalIdRaw
                : parseInt(clickedInternalIdRaw);

            const selectedIdNum = selectedMarkerId !== null
                ? (typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId))
                : null;

            if (selectedIdNum === clickedInternalId) {
                deselectMarker();
            } else {
                selectMarkerInScene(clickedInternalId, clickedMarker);
            }
            markerClicked = true;
        }
    }

    if (!markerClicked) {
        if (placementMode === 'click') {
            if (meshObject) {
                const intersects = raycaster.intersectObject(meshObject, true);
                if (intersects.length > 0) {
                    const point = intersects[0].point;
                    const normal = intersects[0].face.normal;
                    placeMarkerAtPosition([point.x, point.y, point.z], [normal.x, normal.y, normal.z]);
                }
            }
        } else {
            if (meshObject) {
                const intersects = raycaster.intersectObject(meshObject, true);
                if (intersects.length > 0) {
                    deselectMarker();
                    document.getElementById('cadObjectControls').style.display = 'block';
                }
            }
        }
    }
}

function selectMarkerInScene(internalId, markerMesh) {
    if (selectedMarkerMesh && selectedMarkerMesh.material) {
        if (selectedMarkerMesh.userData.originalColor !== undefined) {
            selectedMarkerMesh.material.color.setHex(selectedMarkerMesh.userData.originalColor);
        }
    }

    selectedMarkerId = typeof internalId === 'number' ? internalId : parseInt(internalId);
    selectedMarkerMesh = markerMesh;

    if (markerMesh.userData.originalColor === undefined) {
        markerMesh.userData.originalColor = markerMesh.material.color.getHex();
    }
    markerMesh.material.color.setHex(0x00ff00);

    document.querySelectorAll('.marker-item').forEach(item => {
        item.classList.remove('selected');
        const itemId = parseInt(item.dataset.internalId);
        const compareId = typeof internalId === 'string' ? parseInt(internalId) : internalId;
        if (itemId === compareId || String(itemId) === String(compareId)) {
            item.classList.add('selected');
        }
    });

    document.getElementById('cadObjectControls').style.display = 'none';
    showRotationControls(internalId);
}

function deselectMarker() {
    if (selectedMarkerMesh && selectedMarkerMesh.material) {
        if (selectedMarkerMesh.userData.originalColor !== undefined) {
            selectedMarkerMesh.material.color.setHex(selectedMarkerMesh.userData.originalColor);
        }
    }

    selectedMarkerId = null;
    selectedMarkerMesh = null;

    document.querySelectorAll('.marker-item').forEach(item => {
        item.classList.remove('selected');
    });

    document.getElementById('rotationControls').style.display = 'none';
    document.getElementById('translationControls').style.display = 'none';

    if (meshObject) {
        document.getElementById('cadObjectControls').style.display = 'block';
    }
}

async function loadModel() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/load-model', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            displayModel(data);
            session_state.current_file = file.name;
            updateStatus('modelStatus', 'Model loaded: ' + file.name, 'success');
            await loadCADPose();
        } else {
            updateStatus('modelStatus', 'Error: ' + (data.detail || data.message || 'Unknown error'), 'error');
        }
    } catch (error) {
        updateStatus('modelStatus', 'Error loading model: ' + error.message, 'error');
    }
}

function displayModel(data) {
    markers.forEach(marker => {
        if (marker.parent) {
            marker.parent.remove(marker);
        }
    });
    markers = [];
    selectedMarkerId = null;
    selectedMarkerMesh = null;

    if (meshObject) {
        scene.remove(meshObject);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
    geometry.setIndex(data.faces);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
        color: 0x888888,
        side: THREE.DoubleSide,
        wireframe: false
    });

    meshObject = new THREE.Mesh(geometry, material);
    scene.add(meshObject);

    refreshMarkers().catch(err => console.error('Error refreshing markers:', err));
    document.getElementById('cadObjectControls').style.display = 'block';
    resetCADPose();

    const box = new THREE.Box3().setFromObject(meshObject);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const distance = maxDim * 2;

    camera.position.set(center.x + distance, center.y - distance, center.z + distance);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

async function updateCADPose() {
    if (!meshObject) return;

    const posX = parseFloat(document.getElementById('cadPosX').value) || 0;
    const posY = parseFloat(document.getElementById('cadPosY').value) || 0;
    const posZ = parseFloat(document.getElementById('cadPosZ').value) || 0;

    const roll = parseFloat(document.getElementById('cadRotRoll').value) || 0;
    const pitch = parseFloat(document.getElementById('cadRotPitch').value) || 0;
    const yaw = parseFloat(document.getElementById('cadRotYaw').value) || 0;

    meshObject.position.set(posX, posY, posZ);

    const rollRad = THREE.MathUtils.degToRad(roll);
    const pitchRad = THREE.MathUtils.degToRad(pitch);
    const yawRad = THREE.MathUtils.degToRad(yaw);
    meshObject.rotation.set(rollRad, pitchRad, yawRad);

    updateCADPoseDisplay(posX, posY, posZ, roll, pitch, yaw);

    try {
        await fetch('/api/cad-pose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                position: { x: posX, y: posY, z: posZ },
                rotation: { roll: roll, pitch: pitch, yaw: yaw }
            })
        });
    } catch (error) {
        console.error('Error updating CAD pose:', error);
    }
}

function updateCADPoseDisplay(x, y, z, roll, pitch, yaw) {
    document.getElementById('cadCurrentPos').textContent = `(${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)})`;
    document.getElementById('cadCurrentRPY').textContent = `(${roll.toFixed(1)}, ${pitch.toFixed(1)}, ${yaw.toFixed(1)})`;

    const rollRad = THREE.MathUtils.degToRad(roll);
    const pitchRad = THREE.MathUtils.degToRad(pitch);
    const yawRad = THREE.MathUtils.degToRad(yaw);
    const euler = new THREE.Euler(rollRad, pitchRad, yawRad, 'XYZ');
    const quat = new THREE.Quaternion().setFromEuler(euler);
    document.getElementById('cadCurrentQuat').textContent = `(${quat.x.toFixed(4)}, ${quat.y.toFixed(4)}, ${quat.z.toFixed(4)}, ${quat.w.toFixed(4)})`;

    const angle = Math.acos(Math.max(-1, Math.min(1, Math.abs(quat.w)))) * 2;
    const s = Math.sqrt(1 - quat.w * quat.w);
    let axisX = 0, axisY = 0, axisZ = 1;
    if (s > 0.0001) {
        axisX = quat.x / s;
        axisY = quat.y / s;
        axisZ = quat.z / s;
    }
    const angleDeg = THREE.MathUtils.radToDeg(angle);
    document.getElementById('cadCurrentAxisAngle').textContent = `[${axisX.toFixed(4)}, ${axisY.toFixed(4)}, ${axisZ.toFixed(4)}] @ ${angleDeg.toFixed(2)}`;
}

async function resetCADPose() {
    if (!meshObject) return;

    document.getElementById('cadPosX').value = '0';
    document.getElementById('cadPosY').value = '0';
    document.getElementById('cadPosZ').value = '0';
    document.getElementById('cadRotRoll').value = '0';
    document.getElementById('cadRotPitch').value = '0';
    document.getElementById('cadRotYaw').value = '0';

    await updateCADPose();
}

async function loadCADPose() {
    try {
        const response = await fetch('/api/cad-pose');
        if (response.ok) {
            const data = await response.json();
            const pos = data.position || { x: 0, y: 0, z: 0 };
            const rot = data.rotation || { roll: 0, pitch: 0, yaw: 0 };

            document.getElementById('cadPosX').value = pos.x;
            document.getElementById('cadPosY').value = pos.y;
            document.getElementById('cadPosZ').value = pos.z;
            document.getElementById('cadRotRoll').value = rot.roll;
            document.getElementById('cadRotPitch').value = rot.pitch;
            document.getElementById('cadRotYaw').value = rot.yaw;

            if (meshObject) {
                meshObject.position.set(pos.x, pos.y, pos.z);
                const rollRad = THREE.MathUtils.degToRad(rot.roll);
                const pitchRad = THREE.MathUtils.degToRad(rot.pitch);
                const yawRad = THREE.MathUtils.degToRad(rot.yaw);
                meshObject.rotation.set(rollRad, pitchRad, yawRad);
                updateCADPoseDisplay(pos.x, pos.y, pos.z, rot.roll, rot.pitch, rot.yaw);
            }
        }
    } catch (error) {
        console.error('Error loading CAD pose:', error);
    }
}

async function enterPlacementMode(mode) {
    placementMode = mode;

    if (mode === 'random') {
        await placeRandomMarker();
    } else if (mode === 'face-picker') {
        await showFacePicker();
    } else if (mode === 'smart') {
        await placeSmartMarker();
    } else if (mode === 'all-6') {
        await placeAll6Faces();
    } else if (mode === 'corner') {
        await placeCornerMarkers();
    } else if (mode === 'single-face') {
        await placeSingleMarkerOnFace();
    } else if (mode === 'manual') {
        await showManualPlacement();
    }
}

function getMarkerConfig() {
    return {
        dictionary: document.getElementById('dictSelect').value,
        aruco_id: parseInt(document.getElementById('markerId').value),
        size: parseFloat(document.getElementById('markerSize').value),
        border_width: parseFloat(document.getElementById('borderWidth').value)
    };
}

function getMaxIdForDict(dictName) {
    const parts = dictName.split('_');
    if (parts.length >= 3) {
        return parseInt(parts[parts.length - 1]) - 1;
    }
    return 49;
}

async function placeMarkerAtPosition(position, normal) {
    const config = getMarkerConfig();
    config.position = { x: position[0], y: position[1], z: position[2] };
    config.normal = { x: normal[0], y: normal[1], z: normal[2] };

    try {
        const response = await fetch('/api/add-marker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (response.ok) {
            await addMarkerToScene(data);
            await refreshMarkers();
            placementMode = null;
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function addMarkerToScene(markerData) {
    const pos = markerData.pose_absolute.position;
    const size = markerData.size || 0.021;
    const dictionary = markerData.dictionary || 'DICT_4X4_50';
    const arucoId = markerData.aruco_id;

    const thickness = size * 0.02;
    const geometry = new THREE.BoxGeometry(size, size, thickness);

    const textureLoader = new THREE.TextureLoader();
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
                    console.warn('Failed to load marker texture, using fallback:', error);
                    resolve(null);
                }
            );
        });

        const material = new THREE.MeshStandardMaterial({
            map: texture || null,
            color: texture ? 0xffffff : 0x00ff00,
            side: THREE.DoubleSide,
            transparent: false,
            opacity: 1.0,
            depthWrite: true,
            depthTest: true
        });

        const marker = new THREE.Mesh(geometry, material);
        const originalColor = texture ? 0xffffff : 0x00ff00;
        marker.userData.originalColor = originalColor;

        const offsetDistance = thickness * 0.6;
        const offset = new THREE.Vector3(0, 0, offsetDistance);

        const quat = markerData.pose_absolute.rotation.quaternion;
        marker.quaternion.set(quat.x, quat.y, quat.z, quat.w);

        const worldPos = new THREE.Vector3(pos.x, pos.y, pos.z);
        offset.applyQuaternion(marker.quaternion);

        if (meshObject && (meshObject.position.lengthSq() > 1e-6 || meshObject.quaternion.w !== 1)) {
            const worldToLocal = new THREE.Matrix4().copy(meshObject.matrixWorld).invert();
            const localPos = worldPos.clone().applyMatrix4(worldToLocal);
            marker.position.copy(localPos).add(offset);
        } else {
            marker.position.copy(worldPos).add(offset);
        }

        marker.userData.markerId = arucoId;
        marker.userData.internalId = typeof markerData.internal_id === 'number'
            ? markerData.internal_id
            : parseInt(markerData.internal_id);

        if (meshObject) {
            meshObject.add(marker);
        } else {
            scene.add(marker);
        }
        markers.push(marker);
    } catch (error) {
        console.error('Error creating marker:', error);
    }
}

async function refreshMarkers(skipRotationControlsUpdate = false, skipTranslationUpdate = false) {
    try {
        const previousSelectedId = selectedMarkerId;

        const response = await fetch('/api/markers');
        const data = await response.json();

        markers.forEach(m => {
            if (m.parent) {
                m.parent.remove(m);
            }
        });
        markers = [];
        selectedMarkerMesh = null;

        for (const marker of data.markers) {
            await addMarkerToScene(marker);
        }

        if (previousSelectedId !== null) {
            const prevIdNum = typeof previousSelectedId === 'string' ? parseInt(previousSelectedId) : previousSelectedId;
            const markerMesh = markers.find(m => {
                const meshId = m.userData.internalId;
                const meshIdNum = typeof meshId === 'string' ? parseInt(meshId) : meshId;
                return meshIdNum === prevIdNum;
            });
            if (markerMesh) {
                selectedMarkerId = prevIdNum;
                selectedMarkerMesh = markerMesh;
                if (markerMesh.userData.originalColor !== undefined) {
                    markerMesh.material.color.setHex(0x00ff00);
                }
                document.querySelectorAll('.marker-item').forEach(item => {
                    item.classList.remove('selected');
                    const itemId = parseInt(item.dataset.internalId);
                    if (itemId === prevIdNum || String(itemId) === String(prevIdNum)) {
                        item.classList.add('selected');
                    }
                });

                if (!skipRotationControlsUpdate) {
                    showRotationControls(previousSelectedId, skipTranslationUpdate);
                }
            } else {
                selectedMarkerId = null;
            }
        }

        updateMarkerList(data.markers);
    } catch (error) {
        console.error('Error refreshing markers:', error);
    }
}

function updateMarkerList(markers) {
    const list = document.getElementById('markersList');
    list.innerHTML = '';

    markers.forEach(marker => {
        if (marker.internal_id === undefined || marker.internal_id === null) {
            console.error('Marker missing internal_id:', marker);
            return;
        }
        const internalId = marker.internal_id;
        const item = document.createElement('div');
        item.className = 'marker-item';
        item.textContent = `ArUco ${marker.aruco_id} (${marker.face_type})`;
        item.dataset.internalId = String(internalId);
        item.onclick = (e) => {
            selectMarker(internalId, e);
        };
        list.appendChild(item);
    });

    if (typeof updateSwapDisplay === 'function') {
        updateSwapDisplay().catch(() => {});
    }
}

async function selectMarker(internalId, event) {
    const internalIdNum = typeof internalId === 'string' ? parseInt(internalId) : internalId;

    const markerMesh = markers.find(m => {
        const meshInternalId = m.userData.internalId;
        return (typeof meshInternalId === 'number' ? meshInternalId : parseInt(meshInternalId)) === internalIdNum;
    });

    if (markerMesh) {
        selectMarkerInScene(internalIdNum, markerMesh);
    }
}

async function showRotationControls(internalId, skipTranslationUpdate = false) {
    const rotationPanel = document.getElementById('rotationControls');
    const translationPanel = document.getElementById('translationControls');
    rotationPanel.style.display = 'block';
    translationPanel.style.display = 'block';

    try {
        const response = await fetch('/api/markers');
        const data = await response.json();
        const marker = data.markers.find(m => (m.internal_id || m.aruco_id) === internalId);

        if (marker) {
            const rot = marker.pose_absolute.rotation;
            const rollDeg = snapTo5Degrees(rot.roll * 180 / Math.PI);
            const pitchDeg = snapTo5Degrees(rot.pitch * 180 / Math.PI);
            let yawDeg;
            if (marker.in_plane_rotation_deg !== undefined && marker.in_plane_rotation_deg !== null) {
                yawDeg = snapTo5Degrees(marker.in_plane_rotation_deg);
            } else {
                yawDeg = snapTo5Degrees(rot.yaw * 180 / Math.PI);
            }

            document.getElementById('rotateRoll').value = rollDeg;
            document.getElementById('rotatePitch').value = pitchDeg;
            document.getElementById('rotateYaw').value = yawDeg;

            updateRotationDisplay('roll', rollDeg);
            updateRotationDisplay('pitch', pitchDeg);
            updateRotationDisplay('yaw', yawDeg);

            if (!skipTranslationUpdate) {
                let transX = 0, transY = 0;
                if (marker.translation_offset !== undefined) {
                    transX = marker.translation_offset.x || 0;
                    transY = marker.translation_offset.y || 0;
                }
                document.getElementById('translateX').value = transX.toFixed(4);
                document.getElementById('translateY').value = transY.toFixed(4);
            }
        }
    } catch (error) {
        console.error('Error loading marker rotation:', error);
    }
}

function snapTo5Degrees(value) {
    return Math.round(parseFloat(value) / 5) * 5;
}

function updateRotationDisplay(axis, value, skipSliderUpdate = false) {
    const valueSpan = document.getElementById(axis + 'Value');
    if (valueSpan) {
        const snapped = snapTo5Degrees(value);
        valueSpan.textContent = snapped + '\u00B0';
    }
}

function updateTranslationDisplay(axis, value) {
    // Placeholder for compatibility
}

async function applyRotation() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    const yawSlider = document.getElementById('rotateYaw');
    if (!yawSlider) {
        alert('Yaw slider not found');
        return;
    }

    const yawSliderValue = parseFloat(yawSlider.value) || 0;
    const yaw = snapTo5Degrees(yawSliderValue);

    try {
        const markersResponse = await fetch('/api/markers');
        const markersData = await markersResponse.json();

        const selectedIdNum = typeof selectedMarkerId === 'string' ? parseInt(selectedMarkerId) : selectedMarkerId;
        const marker = markersData.markers.find(m => {
            const markerInternalId = typeof m.internal_id === 'string' ? parseInt(m.internal_id) : m.internal_id;
            return markerInternalId === selectedIdNum;
        });

        if (!marker) {
            alert('Marker not found');
            return;
        }

        const currentInPlaneDeg = marker.in_plane_rotation_deg !== undefined ? marker.in_plane_rotation_deg : 0;
        const yawDelta = yaw - currentInPlaneDeg;

        const markerInternalId = typeof marker.internal_id === 'number'
            ? marker.internal_id
            : parseInt(marker.internal_id);

        const response = await fetch(`/api/markers/${markerInternalId}/rotation`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                yaw: yawDelta,
                mode: "relative"
            })
        });

        if (response.ok) {
            await refreshMarkers(true);
        } else {
            const errorData = await response.json();
            alert('Error: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function resetRotationControls() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    try {
        const markersResponse = await fetch('/api/markers');
        const markersData = await markersResponse.json();

        const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
        const marker = markersData.markers.find(m => {
            const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
            return markerId === selectedIdNum;
        });

        if (!marker) {
            alert('Marker not found');
            return;
        }

        const markerInternalId = typeof marker.internal_id === 'number'
            ? marker.internal_id
            : parseInt(marker.internal_id);

        const response = await fetch(`/api/markers/${markerInternalId}/rotation`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'absolute',
                yaw: 0
            })
        });

        if (response.ok) {
            document.getElementById('rotateYaw').value = 0;
            updateRotationDisplay('yaw', 0);
            await refreshMarkers();
        } else {
            const errorData = await response.json();
            alert('Error resetting rotation: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error resetting rotation: ' + error.message);
    }
}

async function applyTranslation() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    const x = parseFloat(document.getElementById('translateX').value) || 0;
    const y = parseFloat(document.getElementById('translateY').value) || 0;

    try {
        const markersResponse = await fetch('/api/markers');
        const markersData = await markersResponse.json();

        const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
        const marker = markersData.markers.find(m => {
            const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
            return markerId === selectedIdNum;
        });

        if (!marker) {
            alert('Marker not found');
            return;
        }

        const markerInternalId = typeof marker.internal_id === 'number'
            ? marker.internal_id
            : parseInt(marker.internal_id);

        const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'absolute',
                x: x,
                y: y
            })
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('translateX').value = data.translation_offset.x.toFixed(4);
            document.getElementById('translateY').value = data.translation_offset.y.toFixed(4);
            await refreshMarkers(false, true);
        } else {
            const errorData = await response.json();
            alert('Error applying translation: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error applying translation: ' + error.message);
    }
}

async function resetTranslationControls() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    try {
        const markersResponse = await fetch('/api/markers');
        const markersData = await markersResponse.json();

        const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
        const marker = markersData.markers.find(m => {
            const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
            return markerId === selectedIdNum;
        });

        if (!marker) {
            alert('Marker not found');
            return;
        }

        const markerInternalId = typeof marker.internal_id === 'number'
            ? marker.internal_id
            : parseInt(marker.internal_id);

        const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'absolute',
                x: 0,
                y: 0
            })
        });

        if (response.ok) {
            document.getElementById('translateX').value = 0;
            document.getElementById('translateY').value = 0;
            await refreshMarkers();
        } else {
            const errorData = await response.json();
            alert('Error resetting translation: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error resetting translation: ' + error.message);
    }
}

async function moveInPlane(direction) {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    try {
        const stepSize = parseFloat(document.getElementById('inplaneStepSize').value) || 0.0005;

        const markersResponse = await fetch('/api/markers');
        const markersData = await markersResponse.json();

        const selectedIdNum = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
        const marker = markersData.markers.find(m => {
            const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
            return markerId === selectedIdNum;
        });

        if (!marker) {
            alert('Marker not found');
            return;
        }

        const markerInternalId = typeof marker.internal_id === 'number'
            ? marker.internal_id
            : parseInt(marker.internal_id);

        let xDelta = 0;
        let yDelta = 0;

        if (direction === 'axis1_neg') {
            xDelta = -stepSize;
        } else if (direction === 'axis1_pos') {
            xDelta = stepSize;
        } else if (direction === 'axis2_neg') {
            yDelta = -stepSize;
        } else if (direction === 'axis2_pos') {
            yDelta = stepSize;
        }

        const response = await fetch(`/api/markers/${markerInternalId}/translation`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: 'relative',
                x: xDelta,
                y: yDelta
            })
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('translateX').value = data.translation_offset.x.toFixed(4);
            document.getElementById('translateY').value = data.translation_offset.y.toFixed(4);
            await refreshMarkers(false, true);
        } else {
            const errorData = await response.json();
            alert('Error moving marker: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error moving marker: ' + error.message);
    }
}

async function removeSelectedMarker() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }

    try {
        const response = await fetch(`/api/markers/${selectedMarkerId}`, { method: 'DELETE' });
        if (response.ok) {
            await refreshMarkers();
            selectedMarkerId = null;
        } else {
            const data = await response.json();
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function clearAllMarkers() {
    if (!confirm('Clear all markers?')) return;

    try {
        const response = await fetch('/api/markers', { method: 'DELETE' });
        if (response.ok) {
            await refreshMarkers();
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function exportAnnotations() {
    try {
        const response = await fetch('/api/export');
        if (!response.ok) {
            const data = await response.json();
            alert('Error: ' + data.detail);
            return;
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const filename = session_state.current_file ?
            session_state.current_file.replace(/\.[^/.]+$/, '') + '_aruco.json' :
            'aruco.json';
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function exportWireframe() {
    try {
        const response = await fetch('/api/export-wireframe');
        if (!response.ok) {
            const data = await response.json();
            alert('Error: ' + data.detail);
            return;
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const filename = session_state.current_file ?
            session_state.current_file.replace(/\.[^/.]+$/, '') + '_wireframe.json' :
            'wireframe.json';
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function exportPNG() {
    const gridWasVisible = gridHelper && gridHelper.visible;
    const axesWasVisible = axesHelper && axesHelper.visible;
    if (gridHelper) gridHelper.visible = false;
    if (axesHelper) axesHelper.visible = false;

    const originalBackground = scene.background;
    scene.background = null;

    const exportRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true
    });
    exportRenderer.setSize(1920, 1080);
    exportRenderer.setClearColor(0x000000, 0);

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
    const filename = session_state.current_file ?
        session_state.current_file.replace(/\.[^/.]+$/, '') + '_scene.png' :
        'scene.png';
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    camera.left = originalLeft;
    camera.right = originalRight;
    camera.top = originalTop;
    camera.bottom = originalBottom;
    camera.updateProjectionMatrix();

    scene.background = originalBackground;

    if (gridHelper) gridHelper.visible = gridWasVisible;
    if (axesHelper) axesHelper.visible = axesWasVisible;

    exportRenderer.dispose();

    alert('PNG exported (1920x1080, transparent background)');
}

async function importAnnotations() {
    const fileInput = document.getElementById('importFile');
    const file = fileInput.files[0];

    try {
        let response;
        let source = '';

        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            response = await fetch('/api/import', {
                method: 'POST',
                body: formData
            });
            source = 'uploaded file';
        } else {
            response = await fetch('/api/import-auto', {
                method: 'POST'
            });
            source = 'data folder';
        }

        if (response.ok) {
            await refreshMarkers();
            await loadCADPose();
            alert(`Annotations imported successfully from ${source}`);
        } else {
            const data = await response.json();
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function updateStatus(id, message, type) {
    const status = document.getElementById(id);
    status.textContent = message;
    status.className = 'status ' + (type || '');
}

async function placeRandomMarker() {
    try {
        const config = getMarkerConfig();
        const response = await fetch('/api/place-marker/random', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (response.ok) {
            await addMarkerToScene(data);
            await refreshMarkers();
            const markerIdInput = document.getElementById('markerId');
            const maxId = getMaxIdForDict(config.dictionary);
            if (parseInt(markerIdInput.value) < maxId) {
                markerIdInput.value = parseInt(markerIdInput.value) + 1;
            }
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function showFacePicker() {
    try {
        const response = await fetch('/api/faces');
        const data = await response.json();

        let faceList = 'Select a face:\n\n';
        data.faces.forEach((face, idx) => {
            faceList += `${idx}: ${face.face_type || 'Face ' + idx} (area: ${face.area.toFixed(4)})\n`;
        });

        const faceIndex = prompt(faceList + '\nEnter face number:');
        if (faceIndex !== null) {
            const idx = parseInt(faceIndex);
            if (idx >= 0 && idx < data.faces.length) {
                const face = data.faces[idx];
                await placeMarkerAtPosition(face.center, face.normal);
            }
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function placeSmartMarker() {
    try {
        const config = getMarkerConfig();
        const response = await fetch('/api/place-marker/smart', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (response.ok) {
            await addMarkerToScene(data);
            await refreshMarkers();
            const markerIdInput = document.getElementById('markerId');
            const maxId = getMaxIdForDict(config.dictionary);
            if (parseInt(markerIdInput.value) < maxId) {
                markerIdInput.value = parseInt(markerIdInput.value) + 1;
            }
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function placeAll6Faces() {
    try {
        const config = getMarkerConfig();
        const response = await fetch('/api/place-marker/all-6', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (response.ok) {
            for (const marker of data.markers) {
                await addMarkerToScene(marker);
            }
            await refreshMarkers();
            const markerIdInput = document.getElementById('markerId');
            markerIdInput.value = parseInt(markerIdInput.value) + data.markers.length;
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function placeCornerMarkers() {
    try {
        const response = await fetch('/api/faces/primary');
        const data = await response.json();

        if (!response.ok) {
            alert('Error: ' + data.detail);
            return;
        }

        let faceList = 'Select a face for corner markers (4 markers will be placed on corners):\n\n';
        data.faces.forEach((face, idx) => {
            faceList += `${idx}: ${face.name} (normal: [${face.normal[0].toFixed(2)}, ${face.normal[1].toFixed(2)}, ${face.normal[2].toFixed(2)}])\n`;
        });

        const faceIndex = prompt(faceList + '\nEnter face number (0-5):');
        if (faceIndex === null) {
            return;
        }

        const idx = parseInt(faceIndex);
        if (idx < 0 || idx >= data.faces.length) {
            alert('Invalid face number. Please select 0-5.');
            return;
        }

        const config = getMarkerConfig();
        config.face_index = idx;

        const cornerResponse = await fetch('/api/place-marker/corner', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const cornerData = await cornerResponse.json();
        if (cornerResponse.ok) {
            for (const marker of cornerData.markers) {
                await addMarkerToScene(marker);
            }
            await refreshMarkers();
            const markerIdInput = document.getElementById('markerId');
            markerIdInput.value = parseInt(markerIdInput.value) + cornerData.markers.length;
        } else {
            alert('Error: ' + cornerData.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function placeSingleMarkerOnFace() {
    try {
        const response = await fetch('/api/faces/primary');
        const data = await response.json();

        if (!response.ok) {
            alert('Error: ' + data.detail);
            return;
        }

        let faceList = 'Select a face for marker placement (1 marker will be placed at center):\n\n';
        data.faces.forEach((face, idx) => {
            faceList += `${idx}: ${face.name} (normal: [${face.normal[0].toFixed(2)}, ${face.normal[1].toFixed(2)}, ${face.normal[2].toFixed(2)}])\n`;
        });

        const faceIndex = prompt(faceList + '\nEnter face number (0-5):');
        if (faceIndex === null) {
            return;
        }

        const idx = parseInt(faceIndex);
        if (idx < 0 || idx >= data.faces.length) {
            alert('Invalid face number. Please select 0-5.');
            return;
        }

        const selectedFace = data.faces[idx];
        await placeMarkerAtPosition(selectedFace.center, selectedFace.normal);
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function showManualPlacement() {
    const x = parseFloat(prompt('X position:') || '0');
    const y = parseFloat(prompt('Y position:') || '0');
    const z = parseFloat(prompt('Z position:') || '0');
    await placeMarkerAtPosition([x, y, z], [0, 0, 1]);
}

// Swap marker functions
async function updateSwapDisplay() {
    const display1 = document.getElementById('swapMarker1Display');
    const display2 = document.getElementById('swapMarker2Display');

    if (swapMarker1Id !== null) {
        try {
            const response = await fetch('/api/markers');
            const data = await response.json();
            const swap1IdNum = typeof swapMarker1Id === 'number' ? swapMarker1Id : parseInt(swapMarker1Id);
            const m = data.markers.find(m => {
                const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                return markerId === swap1IdNum;
            });
            if (m) {
                display1.textContent = `ArUco ${m.aruco_id}`;
            } else {
                display1.textContent = 'Not assigned';
                swapMarker1Id = null;
            }
        } catch (e) {
            display1.textContent = 'Not assigned';
        }
    } else {
        display1.textContent = 'Not assigned';
    }

    if (swapMarker2Id !== null) {
        try {
            const response = await fetch('/api/markers');
            const data = await response.json();
            const swap2IdNum = typeof swapMarker2Id === 'number' ? swapMarker2Id : parseInt(swapMarker2Id);
            const m = data.markers.find(m => {
                const markerId = typeof m.internal_id === 'number' ? m.internal_id : parseInt(m.internal_id);
                return markerId === swap2IdNum;
            });
            if (m) {
                display2.textContent = `ArUco ${m.aruco_id}`;
            } else {
                display2.textContent = 'Not assigned';
                swapMarker2Id = null;
            }
        } catch (e) {
            display2.textContent = 'Not assigned';
        }
    } else {
        display2.textContent = 'Not assigned';
    }
}

function assignToMarker1() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }
    swapMarker1Id = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
    updateSwapDisplay();
}

function assignToMarker2() {
    if (selectedMarkerId === null) {
        alert('Please select a marker first');
        return;
    }
    swapMarker2Id = typeof selectedMarkerId === 'number' ? selectedMarkerId : parseInt(selectedMarkerId);
    updateSwapDisplay();
}

function clearSwapSelection() {
    swapMarker1Id = null;
    swapMarker2Id = null;
    updateSwapDisplay();
}

async function swapMarkerPositions() {
    if (swapMarker1Id === null || swapMarker2Id === null) {
        alert('Please assign both markers first');
        return;
    }

    if (swapMarker1Id === swapMarker2Id) {
        alert('Cannot swap marker with itself');
        return;
    }

    try {
        const response = await fetch('/api/markers/swap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                marker1_id: swapMarker1Id,
                marker2_id: swapMarker2Id
            })
        });

        if (response.ok) {
            await refreshMarkers();
            clearSwapSelection();
        } else {
            const errorData = await response.json();
            alert('Error: ' + errorData.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

// Window resize handler
window.addEventListener('resize', () => {
    const container = document.getElementById('viewer');
    const frustumSize = 0.5;
    const aspect = container.clientWidth / container.clientHeight;
    camera.left = frustumSize * aspect / -2;
    camera.right = frustumSize * aspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = frustumSize / -2;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Dictionary change handler
document.addEventListener('DOMContentLoaded', () => {
    const dictSelect = document.getElementById('dictSelect');
    if (dictSelect) {
        dictSelect.addEventListener('change', function() {
            const dictName = this.value;
            const parts = dictName.split('_');
            const maxId = parts.length >= 3 ? parseInt(parts[parts.length - 1]) - 1 : 49;
            const markerIdInput = document.getElementById('markerId');
            markerIdInput.max = maxId;
            if (parseInt(markerIdInput.value) > maxId) {
                markerIdInput.value = 0;
            }
        });
    }

    // Initialize the scene
    init();
});
