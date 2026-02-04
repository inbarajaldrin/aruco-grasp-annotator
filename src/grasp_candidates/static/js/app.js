// Three.js scene setup
let scene, camera, renderer, controls;
let graspPointsGroup = null;
let currentObjectData = null;
let currentGraspData = null;
let currentGraspCandidates = null;
let selectedGraspPointId = null;  // Selected grasp point ID (e.g., 1, 2, 3)
let selectedGraspId = null;  // Selected candidate ID (e.g., "1_5")
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();

// Colors
const GRASP_COLOR_DEFAULT = 0x00ff00;  // Green
const GRASP_COLOR_SELECTED_POINT = 0xff0000;  // Red for selected grasp point
const GRASP_COLOR_SELECTED_CANDIDATE = 0x3498db;  // Blue for selected candidate
const GRIPPER_COLOR_DEFAULT = 0xffaa00;  // Orange
const GRIPPER_COLOR_SELECTED = 0xff0000;  // Red for selected candidate's gripper
const WIREFRAME_COLOR = 0x888888;  // Gray

function initScene() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Camera - Orthographic for consistent dimensions when zooming
    const container = document.getElementById('viewer-container');
    const frustumSize = 0.5;
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.OrthographicCamera(
        frustumSize * aspect / -2,  // left
        frustumSize * aspect / 2,   // right
        frustumSize / 2,            // top
        frustumSize / -2,           // bottom
        0.001,                      // near
        1000                        // far
    );
    camera.position.set(0.5, -0.5, 0.5);  // Move camera to look at scene from Y-negative
    camera.up.set(0, 0, 1);  // Set Z as up vector
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateProjectionMatrix();

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minZoom = 0.1;
    controls.maxZoom = 50;

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
    document.getElementById('directionSelect').addEventListener('change', onDirectionChanged);
    document.getElementById('executeButton').addEventListener('click', onExecuteButtonClick);
    document.getElementById('openGripperButton').addEventListener('click', onOpenGripperClick);
    document.getElementById('closeGripperButton').addEventListener('click', onCloseGripperClick);
    document.getElementById('moveToSafeHeightButton').addEventListener('click', onMoveToSafeHeightClick);
    document.getElementById('moveHomeButton').addEventListener('click', onMoveHomeClick);

    // Mouse click handler for grasp point selection
    renderer.domElement.addEventListener('click', onGraspPointClick);

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
    const frustumSize = 0.5;
    const aspect = container.clientWidth / container.clientHeight;
    camera.left = frustumSize * aspect / -2;
    camera.right = frustumSize * aspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = frustumSize / -2;
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
        if (!select) {
            console.error('objectSelect element not found!');
            updateStatus('Error: objectSelect element not found');
            return;
        }

        select.innerHTML = '<option value="">-- Select an object --</option>';

        if (objects && Array.isArray(objects) && objects.length > 0) {
            objects.forEach(objName => {
                const option = document.createElement('option');
                option.value = objName;
                // Format object name: remove _scaled70, replace _ with space, capitalize words
                const displayName = objName.replace('_scaled70', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                option.textContent = displayName;
                select.appendChild(option);
            });

            updateStatus(`Loaded ${objects.length} objects`);
            console.log(`Successfully loaded ${objects.length} objects into dropdown`);
        } else {
            updateStatus('No objects found');
            console.warn('No objects returned from API or empty array:', objects);
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

        // Load grasp data (for positions only)
        const graspResponse = await fetch(`/api/grasp-data/${objectName}`);
        if (!graspResponse.ok) {
            throw new Error('Failed to load grasp data');
        }
        currentGraspData = await graspResponse.json();

        // Load grasp candidates (for approach vectors and orientations)
        const candidatesResponse = await fetch(`/api/grasp-candidates/${objectName}`);
        if (!candidatesResponse.ok) {
            throw new Error('Failed to load grasp candidates');
        }
        currentGraspCandidates = await candidatesResponse.json();

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

        // Create group for everything (wireframe, markers, grasp candidates)
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

        // 3. Add grasp candidates (use grasp point positions from grasp data, approach vectors from candidates)
        let totalCandidates = 0;
        if (currentGraspCandidates && currentGraspCandidates.grasp_candidates) {
            // Group candidates by grasp_point_id to get unique grasp points
            const uniqueGraspPoints = new Map();
            currentGraspCandidates.grasp_candidates.forEach(candidate => {
                const gpId = candidate.grasp_point_id;
                if (!uniqueGraspPoints.has(gpId)) {
                    // Find the corresponding grasp point from grasp data for position
                    const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === gpId);
                    if (graspPoint) {
                        uniqueGraspPoints.set(gpId, graspPoint);
                    }
                }
            });

            // Create visualization for each unique grasp point with all its candidates
            uniqueGraspPoints.forEach((graspPoint, gpId) => {
                // Get all candidates for this grasp point
                const candidatesForPoint = currentGraspCandidates.grasp_candidates.filter(c => c.grasp_point_id === gpId);

                candidatesForPoint.forEach((candidate, idx) => {
                    // Create sphere using grasp point position, but use candidate's approach quaternion/vector
                    const candidateData = {
                        id: `${gpId}_${candidate.direction_id}`, // Unique ID: grasp_point_id_direction_id
                        grasp_point_id: gpId,
                        direction_id: candidate.direction_id,
                        position: graspPoint.position, // Use position from grasp data
                        approach_quaternion: candidate.approach_quaternion, // Use approach quaternion from candidate (new format)
                        approach_vector: candidate.approach_vector, // Use approach vector from candidate (old format, fallback)
                        type: graspPoint.type || 'center_point'
                    };

                    const sphere = createGraspPointSphere(candidateData, totalCandidates);
                    graspPointsGroup.add(sphere);
                    totalCandidates++;
                });
            });
        }

        // Objects are always at world center (0,0,0) - no pose transformation
        // Grasp points are relative to CAD center which is at world center
        graspPointsGroup.position.set(0, 0, 0);
        graspPointsGroup.rotation.set(0, 0, 0);

        scene.add(graspPointsGroup);

        // Update grasp point selector
        updateGraspPointSelector();

        updateStatus(`Loaded ${objectName} (${totalCandidates} grasp candidates)`);
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
    // Calculate candidate ID once
    const candidateId = graspPoint.id || `${graspPoint.grasp_point_id}_${graspPoint.direction_id}`;
    const graspPointId = graspPoint.grasp_point_id;

    // Determine color based on selection state
    let color = GRASP_COLOR_DEFAULT;
    if (selectedGraspPointId === graspPointId) {
        color = GRASP_COLOR_SELECTED_POINT;  // Red for selected grasp point
    } else if (selectedGraspId === candidateId) {
        color = GRASP_COLOR_SELECTED_CANDIDATE;  // Blue for selected candidate
    }
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

    // Store metadata (candidateId already declared above)
    sphere.userData = {
        name: `Grasp-${graspPoint.grasp_point_id || graspPoint.id}_${graspPoint.direction_id || ''}`,
        type: 'grasp_point',
        graspPointId: graspPointId,  // Store grasp point ID for filtering
        graspId: graspPoint.grasp_point_id || graspPoint.id,
        directionId: graspPoint.direction_id,
        id: candidateId, // Use the unique candidate ID
        displayName: `Grasp Point ${graspPoint.grasp_point_id || graspPoint.id}${graspPoint.direction_id ? ` - Direction ${graspPoint.direction_id}` : ''}`,
        originalColor: color,
        isClickable: true  // Mark as clickable
    };

    // Add gripper visualization (parallel-jaw gripper)
    const gripperGroup = new THREE.Group();

    // Gripper parameters
    const gripperWidth = 0.008;      // 8mm - distance between jaws
    const gripperLength = 0.012;     // 12mm - length of gripper fingers
    const gripperDepth = 0.004;      // 4mm - depth of gripper fingers
    const gripperThickness = 0.001;  // 1mm - thickness of each jaw
    const approachLength = 0.015;    // 15mm - length of approach indicator

    // Material for gripper - color depends on selection
    const isCandidateSelected = selectedGraspId === candidateId;
    const gripperColor = isCandidateSelected ? GRIPPER_COLOR_SELECTED : GRIPPER_COLOR_DEFAULT;
    const gripperMaterial = new THREE.MeshPhongMaterial({
        color: gripperColor,
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

    // Orient gripper based on approach quaternion (new format) or approach vector (old format)
    // The approach quaternion represents the orientation where Z-axis points in approach direction
    // The gripper should point TOWARDS the grasp point (opposite to approach direction)
    let approachVec = new THREE.Vector3(0, 0, -1); // Default direction

    if (graspPoint.approach_quaternion) {
        // New format: use approach quaternion directly
        const quat = graspPoint.approach_quaternion;
        const approachQuat = new THREE.Quaternion(quat.x, quat.y, quat.z, quat.w);

        // Extract Z-axis direction from quaternion (this is the approach direction)
        const zAxis = new THREE.Vector3(0, 0, 1);
        zAxis.applyQuaternion(approachQuat);

        // Negate to get gripper pointing direction (towards grasp point)
        approachVec = zAxis.negate().normalize();

        // Create quaternion for gripper orientation (gripper Z points down, opposite to approach)
        const gripperZAxis = new THREE.Vector3(0, 0, 1); // Gripper Z in local frame
        const defaultDir = new THREE.Vector3(0, 0, -1); // Default gripper pointing direction
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(defaultDir, approachVec);
        gripperGroup.quaternion.copy(quaternion);
    } else if (graspPoint.approach_vector) {
        // Old format: use approach vector (backward compatibility)
        const approach = graspPoint.approach_vector;
        // Negate the approach vector to point towards the grasp point
        approachVec = new THREE.Vector3(-approach.x, -approach.y, -approach.z).normalize();

        // Default gripper points along -Z axis, rotate to match negated approach vector
        const defaultDir = new THREE.Vector3(0, 0, -1);
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(defaultDir, approachVec);
        gripperGroup.quaternion.copy(quaternion);
    }

    // Position gripper offset from sphere along approach direction
    // Offset distance: 8mm (0.008m) - this is just for visualization
    // The actual 0.115m offset is applied at runtime
    const offsetDistance = 0.008; // 8mm for visualization
    const offsetPosition = approachVec.clone().multiplyScalar(offsetDistance);
    gripperGroup.position.copy(offsetPosition);
    sphere.add(gripperGroup);

    // Store gripper reference in sphere userData for visibility control
    sphere.userData.gripperGroup = gripperGroup;
    sphere.userData.gripperMaterial = gripperMaterial;  // Store for color updates

    // Make gripper group and all its children clickable
    gripperGroup.userData.isGripper = true;
    gripperGroup.traverse((child) => {
        child.userData.isGripper = true;
        child.userData.parentSphere = sphere;  // Reference back to sphere
    });

    return sphere;
}

function onGraspPointClick(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Update raycaster
    raycaster.setFromCamera(mouse, camera);

    // Find intersections with grasp points
    const intersects = raycaster.intersectObjects(graspPointsGroup.children, true);

    if (intersects.length > 0) {
        // Find the closest grasp point sphere (not gripper parts)
        for (let intersect of intersects) {
            let obj = intersect.object;
            // Traverse up to find the sphere
            while (obj && obj.userData) {
                if (obj.userData.type === 'grasp_point' && obj.userData.isClickable) {
                    const clickedCandidateId = obj.userData.id;  // Full candidate ID (e.g., "6_16")
                    const clickedGraspPointId = obj.userData.graspPointId;
                    const clickedDirectionId = obj.userData.directionId;

                    // If no grasp point is selected, select the grasp point
                    if (selectedGraspPointId === null) {
                        // Select the clicked grasp point
                        selectedGraspPointId = clickedGraspPointId;
                        selectedGraspId = null;  // Clear candidate selection

                        // Update dropdowns
                        document.getElementById('graspSelect').value = clickedGraspPointId.toString();
                        document.getElementById('directionSelect').value = '';
                        document.getElementById('directionSelect').disabled = false;
                        updateDirectionSelector(clickedGraspPointId.toString());

                        // Update visualization
                        updateVisualization();

                        return;
                    }

                    // If a grasp point is already selected, check if clicking on a candidate from that grasp point
                    if (selectedGraspPointId === clickedGraspPointId) {
                        // If clicking the same candidate, deselect it
                        if (selectedGraspId === clickedCandidateId) {
                            selectedGraspId = null;
                            document.getElementById('directionSelect').value = '';
                            updateVisualization();
                            updateGraspInfo(null);
                            // Hide execute button
                            const executeButtonGroup = document.getElementById('executeButtonGroup');
                            const executeButton = document.getElementById('executeButton');
                            if (executeButtonGroup) executeButtonGroup.style.display = 'none';
                            if (executeButton) executeButton.disabled = true;
                            return;
                        }

                        // Select the clicked candidate from the selected grasp point
                        selectedGraspId = clickedCandidateId;

                        // Update dropdowns to reflect the selected candidate
                        document.getElementById('directionSelect').value = clickedDirectionId.toString();

                        // Update grasp info and execute button
                        const candidate = currentGraspCandidates.grasp_candidates.find(
                            c => c.grasp_point_id === clickedGraspPointId && c.direction_id === clickedDirectionId
                        );
                        const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === clickedGraspPointId);

                        if (candidate && graspPoint) {
                            const candidateData = {
                                id: clickedCandidateId,
                                grasp_point_id: clickedGraspPointId.toString(),
                                direction_id: clickedDirectionId.toString(),
                                position: graspPoint.position,
                                approach_quaternion: candidate.approach_quaternion,
                                approach_vector: candidate.approach_vector,
                                type: graspPoint.type || 'center_point'
                            };
                            updateGraspInfo(candidateData);

                            // Show and enable execute button
                            const executeButtonGroup = document.getElementById('executeButtonGroup');
                            const executeButton = document.getElementById('executeButton');
                            if (executeButtonGroup) executeButtonGroup.style.display = 'block';
                            if (executeButton) executeButton.disabled = false;
                        }

                        // Update visualization
                        updateVisualization();

                        return;
                    } else {
                        // Clicking on a different grasp point - select that grasp point instead
                        selectedGraspPointId = clickedGraspPointId;
                        selectedGraspId = null;  // Clear candidate selection

                        // Update dropdowns
                        document.getElementById('graspSelect').value = clickedGraspPointId.toString();
                        document.getElementById('directionSelect').value = '';
                        document.getElementById('directionSelect').disabled = false;
                        updateDirectionSelector(clickedGraspPointId.toString());

                        // Update visualization
                        updateVisualization();

                        return;
                    }
                }
                obj = obj.parent;
            }
        }
    }
}

function clearGraspPoints() {
    if (graspPointsGroup) {
        scene.remove(graspPointsGroup);
        graspPointsGroup = null;
    }
}

function updateGraspPointSelector() {
    const graspSelect = document.getElementById('graspSelect');
    const directionSelect = document.getElementById('directionSelect');

    graspSelect.innerHTML = '<option value="">-- Select a grasp point --</option>';
    directionSelect.innerHTML = '<option value="">-- Select grasp point first --</option>';
    directionSelect.disabled = true;

    if (!currentGraspCandidates || !currentGraspCandidates.grasp_candidates) {
        graspSelect.disabled = true;
        return;
    }

    graspSelect.disabled = false;

    // Get unique grasp point IDs
    const uniqueGraspPointIds = new Set();
    currentGraspCandidates.grasp_candidates.forEach(candidate => {
        uniqueGraspPointIds.add(candidate.grasp_point_id);
    });

    // Sort and populate grasp point dropdown
    const sortedGraspPointIds = Array.from(uniqueGraspPointIds).sort((a, b) => a - b);
    sortedGraspPointIds.forEach(gpId => {
        const option = document.createElement('option');
        option.value = gpId.toString();
        option.textContent = `Grasp Point ${gpId}`;
        graspSelect.appendChild(option);
    });
}

function updateDirectionSelector(graspPointId) {
    const directionSelect = document.getElementById('directionSelect');
    directionSelect.innerHTML = '<option value="">-- Select a direction --</option>';

    if (!graspPointId || !currentGraspCandidates || !currentGraspCandidates.grasp_candidates) {
        directionSelect.disabled = true;
        return;
    }

    directionSelect.disabled = false;

    // Get all directions for this grasp point
    const directions = new Set();
    currentGraspCandidates.grasp_candidates.forEach(candidate => {
        if (candidate.grasp_point_id === parseInt(graspPointId)) {
            directions.add(candidate.direction_id);
        }
    });

    // Sort and populate direction dropdown
    const sortedDirections = Array.from(directions).sort((a, b) => a - b);
    sortedDirections.forEach(dirId => {
        const option = document.createElement('option');
        option.value = dirId.toString();
        option.textContent = `Direction ${dirId}`;
        directionSelect.appendChild(option);
    });
}

function onGraspPointChanged(event) {
    const graspPointId = event.target.value;

    // Clear direction selection when grasp point changes
    const directionSelect = document.getElementById('directionSelect');
    directionSelect.value = '';
    selectedGraspId = null;

    if (graspPointId) {
        selectedGraspPointId = parseInt(graspPointId);
    } else {
        selectedGraspPointId = null;
    }

    // Update direction selector with available directions for this grasp point
    if (graspPointId) {
        updateDirectionSelector(graspPointId);
    } else {
        updateDirectionSelector(null);
    }

    // Update visualization
    updateVisualization();
}

function onDirectionChanged(event) {
    const directionId = event.target.value;
    const graspPointId = document.getElementById('graspSelect').value;

    if (!graspPointId || !directionId) {
        selectedGraspId = null;
        updateVisualization();
        updateGraspInfo(null);
        // Hide execute button
        const executeButtonGroup = document.getElementById('executeButtonGroup');
        const executeButton = document.getElementById('executeButton');
        if (executeButtonGroup) executeButtonGroup.style.display = 'none';
        if (executeButton) executeButton.disabled = true;
        return;
    }

    // Create full candidate ID
    selectedGraspId = `${graspPointId}_${directionId}`;
    const gpId = parseInt(graspPointId);
    const dirId = parseInt(directionId);

    // Find grasp candidate data
    const candidate = currentGraspCandidates.grasp_candidates.find(
        c => c.grasp_point_id === gpId && c.direction_id === dirId
    );

    // Find corresponding grasp point for position
    const graspPoint = currentGraspData.grasp_points.find(gp => gp.id === gpId);

    if (candidate && graspPoint) {
        // Create combined data for display
        const candidateData = {
            id: selectedGraspId,
            grasp_point_id: graspPointId,
            direction_id: directionId,
            position: graspPoint.position,
            approach_quaternion: candidate.approach_quaternion, // New format
            approach_vector: candidate.approach_vector, // Old format (fallback)
            type: graspPoint.type || 'center_point'
        };
        updateGraspInfo(candidateData);

        // Update visualization
        updateVisualization();

        // Show and enable execute button
        const executeButtonGroup = document.getElementById('executeButtonGroup');
        const executeButton = document.getElementById('executeButton');
        if (executeButtonGroup) executeButtonGroup.style.display = 'block';
        if (executeButton) executeButton.disabled = false;
    }
}

function updateVisualization() {
    if (!graspPointsGroup) return;

    graspPointsGroup.traverse((child) => {
        if (child.userData && child.userData.type === 'grasp_point') {
            const graspPointId = child.userData.graspPointId;
            const candidateId = child.userData.id;

            // Determine sphere color
            let sphereColor = GRASP_COLOR_DEFAULT;
            if (selectedGraspPointId === graspPointId) {
                sphereColor = GRASP_COLOR_SELECTED_POINT;  // Red for selected grasp point
            } else if (selectedGraspId === candidateId) {
                sphereColor = GRASP_COLOR_SELECTED_CANDIDATE;  // Blue for selected candidate
            }

            // Update sphere color
            if (child.material) {
                child.material.color.setHex(sphereColor);
                child.material.emissive.setHex(sphereColor);
            }

            // Handle visibility and gripper color based on selection state
            // Priority: candidate selection > grasp point selection > show all
            if (selectedGraspId !== null) {
                // A candidate is selected - only show that candidate, hide all others
                if (candidateId === selectedGraspId) {
                    // Selected candidate: blue sphere, red gripper visible
                    child.visible = true;
                    child.material.opacity = 0.9;

                    // Update gripper to red and make it visible
                    if (child.userData.gripperGroup) {
                        child.userData.gripperGroup.visible = true;
                        if (child.userData.gripperMaterial) {
                            child.userData.gripperMaterial.color.setHex(GRIPPER_COLOR_SELECTED);
                        }
                        child.userData.gripperGroup.traverse((gripperChild) => {
                            if (gripperChild.material) {
                                gripperChild.material.color.setHex(GRIPPER_COLOR_SELECTED);
                            }
                        });
                    }
                } else {
                    // Hide all other candidates (including from same grasp point)
                    child.visible = false;
                }
            } else if (selectedGraspPointId !== null) {
                // A grasp point is selected (but no candidate selected)
                // Show all candidates from selected grasp point, hide others
                if (graspPointId === selectedGraspPointId) {
                    // Show all candidates from selected grasp point
                    child.visible = true;
                    child.material.opacity = 0.9;

                    // Update gripper color - orange for all (no candidate selected)
                    if (child.userData.gripperGroup) {
                        child.userData.gripperGroup.visible = true;
                        const gripperColor = GRIPPER_COLOR_DEFAULT;

                        // Update gripper material color
                        if (child.userData.gripperMaterial) {
                            child.userData.gripperMaterial.color.setHex(gripperColor);
                        }

                        // Update all gripper children colors
                        child.userData.gripperGroup.traverse((gripperChild) => {
                            if (gripperChild.material) {
                                gripperChild.material.color.setHex(gripperColor);
                            }
                        });
                    }
                } else {
                    // Hide candidates from other grasp points
                    child.visible = false;
                }
            } else {
                // Nothing selected: show all with default settings
                child.visible = true;
                child.material.opacity = 0.9;

                if (child.userData.gripperGroup) {
                    child.userData.gripperGroup.visible = true;
                    if (child.userData.gripperMaterial) {
                        child.userData.gripperMaterial.color.setHex(GRIPPER_COLOR_DEFAULT);
                    }
                    child.userData.gripperGroup.traverse((gripperChild) => {
                        if (gripperChild.material) {
                            gripperChild.material.color.setHex(GRIPPER_COLOR_DEFAULT);
                        }
                    });
                }
            }
        }
    });
}

function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

function onExecuteButtonClick() {
    // Execute grasp when button is clicked
    if (selectedGraspId && currentGraspData) {
        executeGrasp(currentGraspData.object_name, selectedGraspId);
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
        updateStatus(`${result.message}`);
        console.log('Grasp execution started:', result);
    } catch (error) {
        console.error('Error executing grasp:', error);
        updateStatus(`Error: ${error.message}`);
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
        updateStatus(`${result.message}`);
        console.log(`Gripper ${command} command sent:`, result);
    } catch (error) {
        console.error(`Error ${command === 'open' ? 'opening' : 'closing'} gripper:`, error);
        updateStatus(`Error: ${error.message}`);
    }
}

function onMoveToSafeHeightClick() {
    moveToSafeHeight();
}

function onMoveHomeClick() {
    moveHome();
}

async function moveToSafeHeight() {
    try {
        updateStatus('Moving to safe height...');
        const response = await fetch('/api/move-to-safe-height', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        updateStatus(`${result.message}`);
        console.log('Move to safe height command sent:', result);
    } catch (error) {
        console.error('Error moving to safe height:', error);
        updateStatus(`Error: ${error.message}`);
    }
}

async function moveHome() {
    try {
        updateStatus('Moving to home position...');
        const response = await fetch('/api/move-home', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        updateStatus(`${result.message}`);
        console.log('Move home command sent:', result);
    } catch (error) {
        console.error('Error moving home:', error);
        updateStatus(`Error: ${error.message}`);
    }
}

function updateGraspInfo(candidateData) {
    const infoDiv = document.getElementById('graspInfo');

    if (!candidateData) {
        infoDiv.innerHTML = `
            <div class="info-item">
                <strong>No grasp candidate selected</strong>
                <span>Select a grasp candidate to view details</span>
            </div>
        `;
        return;
    }

    const pos = candidateData.position;
    const approachQuat = candidateData.approach_quaternion;
    const approachVec = candidateData.approach_vector || { x: 0, y: 0, z: 1 };

    let approachInfo = '';
    if (approachQuat) {
        approachInfo = `
            <div class="info-item">
                <strong>Approach Quaternion:</strong>
                <span>X: ${approachQuat.x.toFixed(4)}, Y: ${approachQuat.y.toFixed(4)}, Z: ${approachQuat.z.toFixed(4)}, W: ${approachQuat.w.toFixed(4)}</span>
            </div>
        `;
    }
    if (approachVec) {
        approachInfo += `
            <div class="info-item">
                <strong>Approach Vector:</strong>
                <span>X: ${approachVec.x.toFixed(3)}, Y: ${approachVec.y.toFixed(3)}, Z: ${approachVec.z.toFixed(3)}</span>
            </div>
        `;
    }

    infoDiv.innerHTML = `
        <div class="info-item">
            <strong>Grasp Point ID:</strong>
            <span>${candidateData.grasp_point_id || candidateData.id}</span>
        </div>
        <div class="info-item">
            <strong>Direction ID:</strong>
            <span>${candidateData.direction_id || 'N/A'}</span>
        </div>
        <div class="info-item">
            <strong>Position:</strong>
            <span>X: ${pos.x.toFixed(4)}m, Y: ${pos.y.toFixed(4)}m, Z: ${pos.z.toFixed(4)}m</span>
        </div>
        ${approachInfo}
        <div class="info-item">
            <strong>Type:</strong>
            <span>${candidateData.type || 'center_point'}</span>
        </div>
    `;
}

function clearScene() {
    clearGraspPoints();
    currentObjectData = null;
    currentGraspData = null;
    currentGraspCandidates = null;
    selectedGraspPointId = null;
    selectedGraspId = null;

    document.getElementById('graspSelect').innerHTML = '<option value="">-- Select object first --</option>';
    document.getElementById('graspSelect').disabled = true;
    updateGraspInfo(null);
}

function updateStatus(message) {
    document.getElementById('statusBar').textContent = message;
}

// Initialize when page loads
window.addEventListener('load', initScene);
