/**
 * Instruction Manual Builder - Main Application
 * Creates assembly instruction manuals from CAD data
 */

// ============================================================
// Global State
// ============================================================
let scene, camera, renderer, controls;
let componentMeshes = {};      // Three.js objects keyed by component name
let assemblyData = null;       // loaded assembly JSON
let manualConfig = {
    title: 'FMB Assembly 1',
    assemblyFile: null,
    steps: [],
    componentOverview: {
        groupByType: true,           // group same-shape components (e.g. fork_orange + fork_yellow = Fork x2)
        useCADOrientation: false,    // false = assembly orientation, true = original CAD orientation
        items: {}                    // per-component overrides: { "fork_orange": { rotation: {x,y,z}, hidden: false, displayName: "...", count: 1 } }
    }
};
let currentStepIndex = -1;
let currentComponentIndex = -1;  // selected component in left sidebar
const frustumSize = 0.5;

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initScene();
    loadAssemblyList();
});

function initScene() {
    const container = document.getElementById('viewer');
    if (!container) return;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    // Orthographic camera (Z-up convention)
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

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.minZoom = 0.1;
    controls.maxZoom = 50;

    // Light (minimal for wireframe visibility)
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, -10, 10);
    scene.add(dirLight);

    // Resize handler
    window.addEventListener('resize', onWindowResize);

    // Animation loop
    animate();
}

function onWindowResize() {
    const container = document.getElementById('viewer');
    if (!container) return;
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

// ============================================================
// Assembly Loading
// ============================================================
async function loadAssemblyList() {
    try {
        const resp = await fetch('/api/assemblies');
        const assemblies = await resp.json();
        const select = document.getElementById('assemblySelect');
        assemblies.forEach(a => {
            const opt = document.createElement('option');
            opt.value = a.name;
            opt.textContent = a.name;
            select.appendChild(opt);
        });
    } catch (e) {
        showStatus('Failed to load assembly list: ' + e.message, 'error');
    }
}

async function loadAssembly() {
    const name = document.getElementById('assemblySelect').value;
    if (!name) {
        showStatus('Please select an assembly first', 'error');
        return;
    }

    showStatus('Loading assembly...', 'info');

    try {
        // Clear existing
        clearScene();

        // Load assembly config
        const resp = await fetch(`/api/assembly/${name}`);
        assemblyData = await resp.json();
        manualConfig.assemblyFile = name;
        manualConfig.title = document.getElementById('manualTitle').value || name;

        // Load OBJ CAD models for each component
        const loader = new THREE.OBJLoader();
        for (const comp of assemblyData.components) {
            showStatus(`Loading ${comp.name}...`, 'info');
            try {
                await loadComponentOBJ(loader, comp);
            } catch (e) {
                console.warn(`Failed to load OBJ for ${comp.name}: ${e.message}`);
            }
        }

        // Reset steps and component selection
        manualConfig.steps = [];
        currentStepIndex = -1;
        currentComponentIndex = -1;
        updateComponentsList();
        updateStepsList();
        updateStepEditor();
        updateComponentEditor();
        updateStatusBar();

        showStatus(`Loaded assembly "${name}" with ${assemblyData.components.length} components`, 'success');
    } catch (e) {
        showStatus('Failed to load assembly: ' + e.message, 'error');
    }
}

function loadComponentOBJ(loader, comp) {
    return new Promise((resolve, reject) => {
        loader.load(
            `/api/model/${comp.name}`,
            (obj) => {
                // OBJ files are in centimeters - scale to meters
                // and center at origin (matching wireframe data convention)
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());

                // Create a group to hold solid mesh + edge lines
                const group = new THREE.Group();
                const edgeThreshold = 15; // degrees - only show edges where faces meet at >15 degrees
                let totalVertices = 0;

                obj.traverse(child => {
                    if (child.isMesh) {
                        const geom = child.geometry.clone();

                        // Center and scale to meters
                        geom.translate(-center.x, -center.y, -center.z);
                        geom.scale(0.01, 0.01, 0.01);
                        geom.computeVertexNormals();

                        totalVertices += geom.attributes.position.count;

                        // Solid white mesh (provides occlusion)
                        const solidMaterial = new THREE.MeshBasicMaterial({
                            color: 0xffffff,
                            side: THREE.FrontSide,
                            polygonOffset: true,
                            polygonOffsetFactor: 1,
                            polygonOffsetUnits: 1,
                        });
                        const solidMesh = new THREE.Mesh(geom, solidMaterial);
                        group.add(solidMesh);

                        // Black edge lines (only hard edges)
                        const edgesGeom = new THREE.EdgesGeometry(geom, edgeThreshold);
                        const edgeMaterial = new THREE.LineBasicMaterial({
                            color: 0x000000,
                            linewidth: 1,
                        });
                        const edgeLines = new THREE.LineSegments(edgesGeom, edgeMaterial);
                        group.add(edgeLines);
                    }
                });

                // Compute geometry fingerprint: vertex count + bounding box dimensions
                // Used to verify components are geometrically identical when grouping
                const scaledBox = new THREE.Box3().setFromObject(group);
                const dims = scaledBox.getSize(new THREE.Vector3());
                const fp = `${totalVertices}_${dims.x.toFixed(4)}_${dims.y.toFixed(4)}_${dims.z.toFixed(4)}`;

                // Apply position and rotation from assembly config
                group.position.set(comp.position.x, comp.position.y, comp.position.z);
                if (comp.rotation && comp.rotation.rpy) {
                    group.rotation.set(
                        comp.rotation.rpy.x,
                        comp.rotation.rpy.y,
                        comp.rotation.rpy.z
                    );
                }

                group.userData = {
                    name: comp.name,
                    type: comp.type,
                    geometryFingerprint: fp,
                    basePosition: { ...comp.position },
                    baseRotation: {
                        x: comp.rotation && comp.rotation.rpy ? comp.rotation.rpy.x : 0,
                        y: comp.rotation && comp.rotation.rpy ? comp.rotation.rpy.y : 0,
                        z: comp.rotation && comp.rotation.rpy ? comp.rotation.rpy.z : 0,
                    }
                };

                scene.add(group);
                componentMeshes[comp.name] = group;
                resolve();
            },
            undefined,
            (error) => reject(error)
        );
    });
}

function clearScene() {
    Object.values(componentMeshes).forEach(group => {
        scene.remove(group);
        group.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    });
    componentMeshes = {};
    assemblyData = null;
}

// ============================================================
// Step Management
// ============================================================
function autoGenerateSteps() {
    if (!assemblyData || !assemblyData.components.length) {
        showStatus('Load an assembly first', 'error');
        return;
    }

    manualConfig.steps = [];

    const hoverHeight = parseFloat(document.getElementById('hoverHeight').value) || 0.05;

    // Sort components by assembly_order (fallback: board first, then file order)
    const ordered = [...assemblyData.components].sort((a, b) => {
        const orderA = a.assembly_order != null ? a.assembly_order : (a.type === 'board' ? 0 : 999);
        const orderB = b.assembly_order != null ? b.assembly_order : (b.type === 'board' ? 0 : 999);
        return orderA - orderB;
    });

    // Create cumulative steps (skip base-only step, start from first insertion)
    for (let i = 1; i < ordered.length; i++) {
        const step = {
            description: `Step ${i}: Add ${ordered[i].name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`,
            components: {},
            camera: null,
            size: 1
        };
        // Show base + all components up to this index; newly added hovers at z=0.05
        assemblyData.components.forEach(comp => {
            const orderIdx = ordered.indexOf(comp);
            step.components[comp.name] = {
                visible: orderIdx <= i,
                offset: { x: 0, y: 0, z: (orderIdx === i) ? hoverHeight : 0 },
                rotation: { x: 0, y: 0, z: 0 }
            };
        });
        manualConfig.steps.push(step);
    }

    // Add "Fully Assembled" as the last step
    // Span full width if preceding steps fill complete rows (even count in 2-col layout)
    const numSteps = manualConfig.steps.length;
    const finalStep = {
        description: 'Fully Assembled',
        components: {},
        camera: null,
        size: (numSteps % 2 === 0) ? 2 : 1
    };
    assemblyData.components.forEach(comp => {
        finalStep.components[comp.name] = { visible: true, offset: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 } };
    });
    manualConfig.steps.push(finalStep);

    selectStep(0);
    updateStepsList();
    showStatus(`Generated ${manualConfig.steps.length} steps`, 'success');
}

function addStep() {
    if (!assemblyData) {
        showStatus('Load an assembly first', 'error');
        return;
    }

    const step = {
        description: `Step ${manualConfig.steps.length + 1}`,
        components: {},
        camera: null,
        size: 1
    };

    assemblyData.components.forEach(comp => {
        step.components[comp.name] = { visible: false, offset: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 } };
    });

    manualConfig.steps.push(step);
    selectStep(manualConfig.steps.length - 1);
    updateStepsList();
}

function deleteStep(index) {
    manualConfig.steps.splice(index, 1);
    if (currentStepIndex >= manualConfig.steps.length) {
        currentStepIndex = manualConfig.steps.length - 1;
    }
    if (currentStepIndex >= 0) {
        selectStep(currentStepIndex);
    } else {
        currentStepIndex = -1;
        updateStepEditor();
    }
    updateStepsList();
}

function moveStepUp(index) {
    if (index <= 0) return;
    const temp = manualConfig.steps[index];
    manualConfig.steps[index] = manualConfig.steps[index - 1];
    manualConfig.steps[index - 1] = temp;
    if (currentStepIndex === index) currentStepIndex = index - 1;
    else if (currentStepIndex === index - 1) currentStepIndex = index;
    updateStepsList();
}

function moveStepDown(index) {
    if (index >= manualConfig.steps.length - 1) return;
    const temp = manualConfig.steps[index];
    manualConfig.steps[index] = manualConfig.steps[index + 1];
    manualConfig.steps[index + 1] = temp;
    if (currentStepIndex === index) currentStepIndex = index + 1;
    else if (currentStepIndex === index + 1) currentStepIndex = index;
    updateStepsList();
}

function selectStep(index) {
    currentStepIndex = index;
    currentComponentIndex = -1;  // deselect component when selecting step
    const step = manualConfig.steps[index];
    if (step) {
        applyStepToScene(step);
        updateStepEditor();
        updateComponentEditor();
        updateStepsList();
        updateComponentsList();
        updateStatusBar();
    }
}

function applyStepToScene(step) {
    for (const [name, config] of Object.entries(step.components)) {
        const mesh = componentMeshes[name];
        if (!mesh) continue;

        mesh.visible = config.visible;

        // Apply XYZ offsets relative to the original assembly position
        // Backward compat: support old zOffset field from saved configs
        const base = mesh.userData.basePosition;
        const offset = config.offset || { x: 0, y: 0, z: config.zOffset || 0 };
        mesh.position.x = base.x + offset.x;
        mesh.position.y = base.y + offset.y;
        mesh.position.z = base.z + offset.z;

        // Apply rotation offset relative to the assembly rotation
        const baseRot = mesh.userData.baseRotation || { x: 0, y: 0, z: 0 };
        const rotOffset = config.rotation || { x: 0, y: 0, z: 0 };
        mesh.rotation.x = baseRot.x + rotOffset.x;
        mesh.rotation.y = baseRot.y + rotOffset.y;
        mesh.rotation.z = baseRot.z + rotOffset.z;
    }

    // Restore camera if step has saved camera
    if (step.camera) {
        camera.position.set(step.camera.position.x, step.camera.position.y, step.camera.position.z);
        camera.zoom = step.camera.zoom;
        camera.updateProjectionMatrix();
        controls.target.set(step.camera.target.x, step.camera.target.y, step.camera.target.z);
        controls.update();
    }
}

// ============================================================
// UI Updates - Components List (Left Sidebar)
// ============================================================
function updateComponentsList() {
    const container = document.getElementById('componentsList');
    if (!assemblyData || !assemblyData.components.length) {
        container.innerHTML = '<div class="empty-state">Load an assembly to begin</div>';
        return;
    }

    const overview = manualConfig.componentOverview || {};
    const groupByType = overview.groupByType !== false;

    if (groupByType) {
        // Build grouped entries: one card per unique geometry
        const groups = {};
        const groupOrder = [];
        assemblyData.components.forEach((comp, i) => {
            const mesh = componentMeshes[comp.name];
            const groupKey = mesh && mesh.userData.geometryFingerprint
                ? mesh.userData.geometryFingerprint
                : comp.name;
            if (!groups[groupKey]) {
                groups[groupKey] = {
                    representative: comp.name,
                    representativeIndex: i,
                    members: [],
                    displayName: formatComponentName(comp.name)
                };
                groupOrder.push(groupKey);
            }
            groups[groupKey].members.push(comp.name);
        });

        container.innerHTML = groupOrder.map((key, gi) => {
            const group = groups[key];
            const repItem = _ensureOverviewItem(group.representative);
            const included = !repItem.hidden;
            const displayName = repItem.displayName || group.displayName;
            // Sum counts across all members
            let totalCount = 0;
            group.members.forEach(name => {
                const item = _ensureOverviewItem(name);
                totalCount += item.count !== undefined ? item.count : 1;
            });
            return `
                <div class="comp-card ${group.representativeIndex === currentComponentIndex ? 'active' : ''} ${!included ? 'excluded' : ''}"
                     onclick="selectComponent(${group.representativeIndex})">
                    <input type="checkbox" ${included ? 'checked' : ''}
                           onclick="event.stopPropagation()" onchange="toggleComponentGroupInclude('${key}', this.checked)">
                    <span class="comp-card-name ${!included ? 'excluded' : ''}">${escapeHtml(displayName)}</span>
                    <input type="number" class="comp-card-count" value="${totalCount}" min="0" step="1"
                           title="Total count for this component type"
                           onclick="event.stopPropagation()" onchange="setComponentGroupCount('${key}', parseInt(this.value) || 1)">
                </div>
            `;
        }).join('');
    } else {
        // Ungrouped: show every individual component
        container.innerHTML = assemblyData.components.map((comp, i) => {
            const item = _ensureOverviewItem(comp.name);
            const included = !item.hidden;
            const displayName = item.displayName || comp.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const count = item.count !== undefined ? item.count : 1;
            return `
                <div class="comp-card ${i === currentComponentIndex ? 'active' : ''} ${!included ? 'excluded' : ''}" onclick="selectComponent(${i})">
                    <input type="checkbox" ${included ? 'checked' : ''}
                           onclick="event.stopPropagation()" onchange="toggleComponentInclude('${comp.name}', this.checked)">
                    <span class="comp-card-name ${!included ? 'excluded' : ''}">${escapeHtml(displayName)}</span>
                    <input type="number" class="comp-card-count" value="${count}" min="0" step="1"
                           title="Number of this component"
                           onclick="event.stopPropagation()" onchange="setComponentCount('${comp.name}', parseInt(this.value) || 1)">
                </div>
            `;
        }).join('');
    }
}

// Build geometry groups lookup (reused by multiple functions)
function _getGeometryGroups() {
    const groups = {};
    if (!assemblyData) return groups;
    assemblyData.components.forEach(comp => {
        const mesh = componentMeshes[comp.name];
        const key = mesh && mesh.userData.geometryFingerprint
            ? mesh.userData.geometryFingerprint
            : comp.name;
        if (!groups[key]) groups[key] = [];
        groups[key].push(comp.name);
    });
    return groups;
}

function toggleComponentGroupInclude(groupKey, included) {
    const groups = _getGeometryGroups();
    const members = groups[groupKey] || [];
    members.forEach(name => {
        const item = _ensureOverviewItem(name);
        item.hidden = !included;
    });
    updateComponentsList();
}

function setComponentGroupCount(groupKey, totalCount) {
    const groups = _getGeometryGroups();
    const members = groups[groupKey] || [];
    // Distribute count evenly across members, remainder to first
    const perMember = Math.floor(totalCount / members.length);
    const remainder = totalCount % members.length;
    members.forEach((name, i) => {
        const item = _ensureOverviewItem(name);
        item.count = perMember + (i < remainder ? 1 : 0);
    });
}

function selectComponent(index) {
    currentComponentIndex = index;
    currentStepIndex = -1;  // deselect step when selecting component
    applyComponentPreview(index);
    updateComponentsList();
    updateStepsList();
    updateStepEditor();
    updateComponentEditor();
    updateStatusBar();
}

function applyComponentPreview(index) {
    if (!assemblyData || index < 0) return;
    const comp = assemblyData.components[index];
    const useCAD = manualConfig.componentOverview && manualConfig.componentOverview.useCADOrientation;

    // Show only the selected component, positioned at origin
    Object.entries(componentMeshes).forEach(([name, mesh]) => {
        if (name === comp.name) {
            mesh.visible = true;
            mesh.position.set(0, 0, 0);
            if (useCAD) {
                mesh.rotation.set(0, 0, 0);
            } else {
                const baseRot = mesh.userData.baseRotation || { x: 0, y: 0, z: 0 };
                mesh.rotation.set(baseRot.x, baseRot.y, baseRot.z);
            }
            // Apply overview rotation override
            const item = manualConfig.componentOverview && manualConfig.componentOverview.items
                ? manualConfig.componentOverview.items[name] : null;
            if (item && item.rotation) {
                mesh.rotation.x += item.rotation.x;
                mesh.rotation.y += item.rotation.y;
                mesh.rotation.z += item.rotation.z;
            }
        } else {
            mesh.visible = false;
        }
    });

    // Fit camera to the component
    const targetMesh = componentMeshes[comp.name];
    if (targetMesh) {
        const box = new THREE.Box3().setFromObject(targetMesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const padding = 1.5;

        camera.position.set(
            center.x + maxDim * 2,
            center.y - maxDim * 2,
            center.z + maxDim * 2
        );
        camera.zoom = frustumSize / (maxDim * padding);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
    }
}

function toggleComponentInclude(name, included) {
    const item = _ensureOverviewItem(name);
    item.hidden = !included;
    updateComponentsList();
}

function setComponentCount(name, count) {
    const item = _ensureOverviewItem(name);
    item.count = Math.max(0, count);
}

// ============================================================
// UI Updates - Component Editor (Right Sidebar)
// ============================================================
function updateComponentEditor() {
    const container = document.getElementById('componentEditorContent');
    if (currentComponentIndex < 0 || !assemblyData) {
        container.innerHTML = '<div class="empty-state">Select a component to edit</div>';
        return;
    }

    const comp = assemblyData.components[currentComponentIndex];
    const item = _ensureOverviewItem(comp.name);
    const displayName = item.displayName || comp.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const rot = item.rotation || { x: 0, y: 0, z: 0 };
    const rotStep = Math.round(Math.PI / 36 * 1000) / 1000; // 5 degrees

    let html = '';

    // Display name
    html += `
        <div class="comp-editor-section">
            <h4>Display Name</h4>
            <input type="text" class="comp-editor-name-input" value="${escapeHtml(displayName)}"
                   onchange="setOverviewItemName('${comp.name}', this.value); updateComponentsList()">
        </div>
    `;

    // Rotation
    html += `
        <div class="comp-editor-section">
            <h4>Orientation</h4>
            <div class="component-offset-row" style="padding-left: 0;">
                <span class="offset-row-label">Rot</span>
                ${['x', 'y', 'z'].map(axis => `
                    <div class="offset-axis-group">
                        <label class="offset-axis-label">${axis.toUpperCase()}</label>
                        <button class="offset-nudge-btn offset-dec" onclick="nudgeOverviewRotation('${comp.name}', '${axis}', -${rotStep})">-</button>
                        <input type="number" class="offset-axis-input" step="0.01"
                               value="${rot[axis].toFixed(3)}"
                               onchange="setOverviewRotation('${comp.name}', '${axis}', parseFloat(this.value))">
                        <button class="offset-nudge-btn offset-inc" onclick="nudgeOverviewRotation('${comp.name}', '${axis}', ${rotStep})">+</button>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// ============================================================
// UI Updates - Steps List & Step Editor
// ============================================================
function updateStepsList() {
    const container = document.getElementById('stepsList');
    if (!manualConfig.steps.length) {
        container.innerHTML = '<div class="empty-state">No steps yet. Load an assembly and auto-generate or add steps.</div>';
        return;
    }

    container.innerHTML = manualConfig.steps.map((step, i) => `
        <div class="step-card ${i === currentStepIndex ? 'active' : ''}" onclick="selectStep(${i})">
            <span class="step-number">${i + 1}</span>
            <span class="step-desc-preview">${escapeHtml(step.description)}</span>
            <select class="step-size-select" title="Image width in export" onclick="event.stopPropagation()" onchange="setStepSize(${i}, parseInt(this.value))">
                <option value="1" ${step.size === 1 ? 'selected' : ''}>Half</option>
                <option value="2" ${step.size === 2 ? 'selected' : ''}>Full</option>
            </select>
            <div class="step-card-actions" onclick="event.stopPropagation()">
                <button onclick="moveStepUp(${i})" title="Move up">&#9650;</button>
                <button onclick="moveStepDown(${i})" title="Move down">&#9660;</button>
                <button class="delete-btn" onclick="deleteStep(${i})" title="Delete">&#10005;</button>
            </div>
        </div>
    `).join('');
}

function updateStepEditor() {
    const container = document.getElementById('stepEditorContent');
    if (currentStepIndex < 0 || !manualConfig.steps[currentStepIndex]) {
        container.innerHTML = '<div class="empty-state">Select a step to edit</div>';
        return;
    }

    // Preserve scroll position across re-renders
    const scrollEl = container.querySelector('.component-visibility-list');
    const savedScroll = scrollEl ? scrollEl.scrollTop : 0;

    const step = manualConfig.steps[currentStepIndex];

    let html = '';

    // Description
    html += `
        <div class="step-editor-section">
            <h4>Description</h4>
            <textarea class="step-description-input"
                      onchange="updateStepDescription(this.value)"
                      oninput="updateStepDescription(this.value)">${escapeHtml(step.description)}</textarea>
        </div>
    `;

    // Component visibility
    html += `<div class="step-editor-section">
        <h4>Components</h4>
        <div style="margin-bottom: 8px;">
            <button class="btn btn-small" onclick="setAllVisible(true)" style="width: auto; display: inline-block; padding: 4px 10px; font-size: 11px;">Show All</button>
            <button class="btn btn-small" onclick="setAllVisible(false)" style="width: auto; display: inline-block; padding: 4px 10px; font-size: 11px;">Hide All</button>
        </div>
        <div class="component-visibility-list">`;

    if (assemblyData) {
        assemblyData.components.forEach(comp => {
            const config = step.components[comp.name] || { visible: false, offset: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 } };
            const offset = config.offset || { x: 0, y: 0, z: config.zOffset || 0 };
            const rot = config.rotation || { x: 0, y: 0, z: 0 };
            const rotStep = Math.round(Math.PI / 36 * 1000) / 1000; // 5 degrees
            html += `
                <div class="component-vis-row">
                    <input type="checkbox"
                           ${config.visible ? 'checked' : ''}
                           onchange="toggleComponentVisibility('${comp.name}', this.checked)">
                    <span class="component-vis-name ${config.visible ? '' : 'hidden'}">${formatComponentName(comp.name)}</span>
                </div>
                <div class="component-offset-row">
                    <span class="offset-row-label">Pos</span>
                    ${['x', 'y', 'z'].map(axis => `
                        <div class="offset-axis-group">
                            <label class="offset-axis-label">${axis.toUpperCase()}</label>
                            <button class="offset-nudge-btn offset-dec" onclick="nudgeComponentOffset('${comp.name}', '${axis}', -0.01)">-</button>
                            <input type="number" class="offset-axis-input" step="0.001"
                                   value="${offset[axis].toFixed(3)}"
                                   onchange="setComponentOffset('${comp.name}', '${axis}', parseFloat(this.value))">
                            <button class="offset-nudge-btn offset-inc" onclick="nudgeComponentOffset('${comp.name}', '${axis}', 0.01)">+</button>
                        </div>
                    `).join('')}
                </div>
                <div class="component-offset-row">
                    <span class="offset-row-label">Rot</span>
                    ${['x', 'y', 'z'].map(axis => `
                        <div class="offset-axis-group">
                            <label class="offset-axis-label">${axis.toUpperCase()}</label>
                            <button class="offset-nudge-btn offset-dec" onclick="nudgeComponentRotation('${comp.name}', '${axis}', -${rotStep})">-</button>
                            <input type="number" class="offset-axis-input" step="0.01"
                                   value="${rot[axis].toFixed(3)}"
                                   onchange="setComponentRotation('${comp.name}', '${axis}', parseFloat(this.value))">
                            <button class="offset-nudge-btn offset-inc" onclick="nudgeComponentRotation('${comp.name}', '${axis}', ${rotStep})">+</button>
                        </div>
                    `).join('')}
                </div>
            `;
        });
    }

    html += '</div></div>';

    // Camera
    html += `
        <div class="step-editor-section">
            <h4>Camera</h4>
            <div class="camera-actions">
                <button class="btn" onclick="saveCameraForStep()">Save Angle</button>
                <button class="btn btn-outline" onclick="resetCamera()">Reset</button>
            </div>
            ${step.camera ? '<span class="camera-saved-badge">Camera angle saved</span>' : ''}
        </div>
    `;

    container.innerHTML = html;

    // Restore scroll position
    const newScrollEl = container.querySelector('.component-visibility-list');
    if (newScrollEl && savedScroll) {
        newScrollEl.scrollTop = savedScroll;
    }
}

function updateStatusBar() {
    const stepInfo = document.getElementById('stepInfo');
    const componentCount = document.getElementById('componentCount');
    const statusText = document.getElementById('statusText');

    if (currentStepIndex >= 0 && manualConfig.steps[currentStepIndex]) {
        stepInfo.textContent = `Step: ${currentStepIndex + 1}/${manualConfig.steps.length}`;
        const visible = Object.values(manualConfig.steps[currentStepIndex].components)
            .filter(c => c.visible).length;
        componentCount.textContent = `Visible: ${visible}/${assemblyData ? assemblyData.components.length : 0}`;
        statusText.textContent = manualConfig.steps[currentStepIndex].description;
    } else {
        stepInfo.textContent = 'Step: --';
        componentCount.textContent = `Components: ${assemblyData ? assemblyData.components.length : 0}`;
        statusText.textContent = assemblyData ? 'Select or create a step' : 'Load an assembly to begin';
    }
}

// ============================================================
// Step Editor Actions
// ============================================================
function updateStepDescription(value) {
    if (currentStepIndex < 0) return;
    manualConfig.steps[currentStepIndex].description = value;
    // Update the step list preview without full re-render to avoid losing focus
    const cards = document.querySelectorAll('.step-card .step-desc-preview');
    if (cards[currentStepIndex]) {
        cards[currentStepIndex].textContent = value;
    }
}

function toggleComponentVisibility(name, visible) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    if (!step.components[name]) {
        step.components[name] = { visible: false, offset: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 } };
    }
    step.components[name].visible = visible;
    applyStepToScene(step);

    // Update name styling
    const rows = document.querySelectorAll('.component-vis-row');
    rows.forEach(row => {
        const nameSpan = row.querySelector('.component-vis-name');
        const checkbox = row.querySelector('input[type="checkbox"]');
        if (nameSpan && checkbox) {
            nameSpan.classList.toggle('hidden', !checkbox.checked);
        }
    });
    updateStatusBar();
}

function _ensureComponent(step, name) {
    if (!step.components[name]) {
        step.components[name] = { visible: false, offset: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 } };
    }
    if (!step.components[name].offset) {
        step.components[name].offset = { x: 0, y: 0, z: step.components[name].zOffset || 0 };
    }
    if (!step.components[name].rotation) {
        step.components[name].rotation = { x: 0, y: 0, z: 0 };
    }
}

function setComponentOffset(name, axis, value) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    _ensureComponent(step, name);
    step.components[name].offset[axis] = value;
    applyStepToScene(step);
}

function nudgeComponentOffset(name, axis, delta) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    _ensureComponent(step, name);
    const newValue = step.components[name].offset[axis] + delta;
    step.components[name].offset[axis] = Math.round(newValue * 1000) / 1000;
    applyStepToScene(step);
    updateStepEditor();
}

function setComponentRotation(name, axis, value) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    _ensureComponent(step, name);
    step.components[name].rotation[axis] = value;
    applyStepToScene(step);
}

function nudgeComponentRotation(name, axis, delta) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    _ensureComponent(step, name);
    const newValue = step.components[name].rotation[axis] + delta;
    step.components[name].rotation[axis] = Math.round(newValue * 1000) / 1000;
    applyStepToScene(step);
    updateStepEditor();
}

function setAllVisible(visible) {
    if (currentStepIndex < 0) return;
    const step = manualConfig.steps[currentStepIndex];
    Object.keys(step.components).forEach(name => {
        step.components[name].visible = visible;
    });
    applyStepToScene(step);
    updateStepEditor();
    updateStatusBar();
}

// ============================================================
// Component Overview Data Helpers
// ============================================================
function _ensureOverviewItem(name) {
    if (!manualConfig.componentOverview) {
        manualConfig.componentOverview = { groupByType: true, useCADOrientation: false, items: {} };
    }
    if (!manualConfig.componentOverview.items) {
        manualConfig.componentOverview.items = {};
    }
    if (!manualConfig.componentOverview.items[name]) {
        manualConfig.componentOverview.items[name] = {};
    }
    return manualConfig.componentOverview.items[name];
}

function setOverviewItemName(name, displayName) {
    const item = _ensureOverviewItem(name);
    item.displayName = displayName;
}

function setOverviewRotation(name, axis, value) {
    const item = _ensureOverviewItem(name);
    if (!item.rotation) item.rotation = { x: 0, y: 0, z: 0 };
    item.rotation[axis] = value;
}

function nudgeOverviewRotation(name, axis, delta) {
    const item = _ensureOverviewItem(name);
    if (!item.rotation) item.rotation = { x: 0, y: 0, z: 0 };
    const newValue = item.rotation[axis] + delta;
    item.rotation[axis] = Math.round(newValue * 1000) / 1000;
    updateComponentEditor();
}

function toggleGroupComponents(grouped) {
    if (!manualConfig.componentOverview) manualConfig.componentOverview = {};
    manualConfig.componentOverview.groupByType = grouped;
    updateComponentsList();
}

function toggleCADOrientation(useCAD) {
    if (!manualConfig.componentOverview) manualConfig.componentOverview = {};
    manualConfig.componentOverview.useCADOrientation = useCAD;
}

function setStepSize(index, size) {
    manualConfig.steps[index].size = size;
}

function saveCameraForStep() {
    if (currentStepIndex < 0) return;
    manualConfig.steps[currentStepIndex].camera = {
        position: {
            x: camera.position.x,
            y: camera.position.y,
            z: camera.position.z
        },
        zoom: camera.zoom,
        target: {
            x: controls.target.x,
            y: controls.target.y,
            z: controls.target.z
        }
    };
    updateStepEditor();
    showStatus('Camera angle saved for this step', 'success');
}

function resetCamera() {
    camera.position.set(0.5, -0.5, 0.5);
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
}

// ============================================================
// Off-Screen Rendering
// ============================================================
let _exportRenderer = null;

function getExportRenderer(width, height) {
    if (!_exportRenderer) {
        _exportRenderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            preserveDrawingBuffer: true
        });
    }
    _exportRenderer.setSize(width, height);
    _exportRenderer.setClearColor(0xffffff, 1);
    return _exportRenderer;
}

function disposeExportRenderer() {
    if (_exportRenderer) {
        _exportRenderer.dispose();
        _exportRenderer = null;
    }
}

function renderStepToImage(step, width, height) {
    return new Promise(resolve => {
        const exportRenderer = getExportRenderer(width, height);

        // Apply step state
        applyStepToScene(step);

        const exportAspect = width / height;
        const exportCamera = new THREE.OrthographicCamera(
            -1, 1, 1, -1, 0.001, 1000
        );
        exportCamera.up.set(0, 0, 1);

        if (step.camera) {
            // Use saved camera state
            const camFrustum = frustumSize;
            exportCamera.left = camFrustum * exportAspect / -2;
            exportCamera.right = camFrustum * exportAspect / 2;
            exportCamera.top = camFrustum / 2;
            exportCamera.bottom = camFrustum / -2;
            exportCamera.position.set(step.camera.position.x, step.camera.position.y, step.camera.position.z);
            exportCamera.zoom = step.camera.zoom;
            const target = step.camera.target;
            exportCamera.lookAt(target.x, target.y, target.z);
        } else {
            // Auto-fit camera to visible objects
            const visibleBox = new THREE.Box3();
            let hasVisible = false;
            Object.entries(step.components).forEach(([name, config]) => {
                if (config.visible && componentMeshes[name]) {
                    visibleBox.expandByObject(componentMeshes[name]);
                    hasVisible = true;
                }
            });

            if (hasVisible) {
                const center = visibleBox.getCenter(new THREE.Vector3());
                const size = visibleBox.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);

                // Set camera direction first so we can project the bounding box
                exportCamera.position.set(
                    center.x + maxDim * 2,
                    center.y - maxDim * 2,
                    center.z + maxDim * 2
                );
                exportCamera.lookAt(center.x, center.y, center.z);
                exportCamera.updateMatrixWorld();

                // Project all 8 corners of the bounding box into camera view space
                const corners = [
                    new THREE.Vector3(visibleBox.min.x, visibleBox.min.y, visibleBox.min.z),
                    new THREE.Vector3(visibleBox.min.x, visibleBox.min.y, visibleBox.max.z),
                    new THREE.Vector3(visibleBox.min.x, visibleBox.max.y, visibleBox.min.z),
                    new THREE.Vector3(visibleBox.min.x, visibleBox.max.y, visibleBox.max.z),
                    new THREE.Vector3(visibleBox.max.x, visibleBox.min.y, visibleBox.min.z),
                    new THREE.Vector3(visibleBox.max.x, visibleBox.min.y, visibleBox.max.z),
                    new THREE.Vector3(visibleBox.max.x, visibleBox.max.y, visibleBox.min.z),
                    new THREE.Vector3(visibleBox.max.x, visibleBox.max.y, visibleBox.max.z),
                ];
                const viewMatrix = exportCamera.matrixWorldInverse;
                let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
                corners.forEach(c => {
                    const projected = c.clone().applyMatrix4(viewMatrix);
                    minX = Math.min(minX, projected.x);
                    maxX = Math.max(maxX, projected.x);
                    minY = Math.min(minY, projected.y);
                    maxY = Math.max(maxY, projected.y);
                });
                const projectedWidth = maxX - minX;
                const projectedHeight = maxY - minY;

                const padding = 1.3;
                const camSizeX = projectedWidth * padding;
                const camSizeY = projectedHeight * padding;
                const camSize = Math.max(camSizeY, camSizeX / exportAspect);

                exportCamera.left = camSize * exportAspect / -2;
                exportCamera.right = camSize * exportAspect / 2;
                exportCamera.top = camSize / 2;
                exportCamera.bottom = camSize / -2;
            } else {
                // Fallback to current camera
                exportCamera.left = frustumSize * exportAspect / -2;
                exportCamera.right = frustumSize * exportAspect / 2;
                exportCamera.top = frustumSize / 2;
                exportCamera.bottom = frustumSize / -2;
                exportCamera.position.copy(camera.position);
                exportCamera.zoom = camera.zoom;
                exportCamera.lookAt(controls.target.x, controls.target.y, controls.target.z);
            }
        }
        exportCamera.updateProjectionMatrix();

        exportRenderer.render(scene, exportCamera);
        const dataURL = exportRenderer.domElement.toDataURL('image/png');

        resolve(dataURL);
    });
}

function renderComponentAlone(componentName, width, height, useCADOrientation = false) {
    return new Promise(resolve => {
        // Save all mesh states
        const savedStates = {};
        Object.entries(componentMeshes).forEach(([name, mesh]) => {
            savedStates[name] = {
                visible: mesh.visible,
                positionX: mesh.position.x,
                positionY: mesh.position.y,
                positionZ: mesh.position.z,
                rotationX: mesh.rotation.x,
                rotationY: mesh.rotation.y,
                rotationZ: mesh.rotation.z
            };
        });

        // Hide all except target
        Object.entries(componentMeshes).forEach(([name, mesh]) => {
            mesh.visible = (name === componentName);
        });

        // Temporarily move target to origin and reset rotation for clean render
        const targetMesh = componentMeshes[componentName];
        if (!targetMesh) {
            // Restore and return empty
            Object.entries(componentMeshes).forEach(([name, mesh]) => {
                const s = savedStates[name];
                mesh.visible = s.visible;
                mesh.position.set(s.positionX, s.positionY, s.positionZ);
                mesh.rotation.set(s.rotationX, s.rotationY, s.rotationZ);
            });
            resolve(null);
            return;
        }

        targetMesh.position.set(0, 0, 0);
        // Reset rotation to CAD original if useCADOrientation is true
        if (useCADOrientation) {
            targetMesh.rotation.set(0, 0, 0);
        }

        // Apply per-component overview rotation override if set
        const overview = manualConfig.componentOverview || {};
        const itemOverride = overview.items && overview.items[componentName];
        if (itemOverride && itemOverride.rotation) {
            targetMesh.rotation.x += itemOverride.rotation.x;
            targetMesh.rotation.y += itemOverride.rotation.y;
            targetMesh.rotation.z += itemOverride.rotation.z;
        }

        // Reuse shared off-screen renderer
        const exportRenderer = getExportRenderer(width, height);

        // Fit camera to component by projecting bounding box into camera space
        const box = new THREE.Box3().setFromObject(targetMesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        // Set up camera direction first so we can project the box
        const exportCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 1000);
        exportCamera.up.set(0, 0, 1);
        exportCamera.position.set(
            center.x + maxDim * 2,
            center.y - maxDim * 2,
            center.z + maxDim * 2
        );
        exportCamera.lookAt(center.x, center.y, center.z);
        exportCamera.updateMatrixWorld();

        // Project all 8 corners of the bounding box into camera view space
        const corners = [
            new THREE.Vector3(box.min.x, box.min.y, box.min.z),
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.max.z),
        ];
        const viewMatrix = exportCamera.matrixWorldInverse;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        corners.forEach(c => {
            const projected = c.clone().applyMatrix4(viewMatrix);
            minX = Math.min(minX, projected.x);
            maxX = Math.max(maxX, projected.x);
            minY = Math.min(minY, projected.y);
            maxY = Math.max(maxY, projected.y);
        });
        const projectedWidth = maxX - minX;
        const projectedHeight = maxY - minY;

        const exportAspect = width / height;
        const padding = 1.4;
        // Use the projected dimensions to determine frustum size
        const camSizeX = projectedWidth * padding;
        const camSizeY = projectedHeight * padding;
        // Pick the larger of aspect-corrected dimensions
        const camSize = Math.max(camSizeY, camSizeX / exportAspect);
        exportCamera.left = camSize * exportAspect / -2;
        exportCamera.right = camSize * exportAspect / 2;
        exportCamera.top = camSize / 2;
        exportCamera.bottom = camSize / -2;
        exportCamera.updateProjectionMatrix();

        exportRenderer.render(scene, exportCamera);
        const dataURL = exportRenderer.domElement.toDataURL('image/png');

        // Restore all mesh states
        Object.entries(componentMeshes).forEach(([name, mesh]) => {
            const s = savedStates[name];
            mesh.visible = s.visible;
            mesh.position.set(s.positionX, s.positionY, s.positionZ);
            mesh.rotation.set(s.rotationX, s.rotationY, s.rotationZ);
        });

        resolve(dataURL);
    });
}

// ============================================================
// Component Grouping for Export
// ============================================================
function getUniqueComponentsWithCounts() {
    if (!assemblyData) return [];

    const overview = manualConfig.componentOverview || {};
    const groupByType = overview.groupByType !== false;
    const groups = {};

    assemblyData.components.forEach(comp => {
        // Skip if explicitly hidden in overview
        const itemData = overview.items && overview.items[comp.name];
        if (itemData && itemData.hidden) {
            return;
        }

        // Determine grouping key: geometry fingerprint when grouping, else component name
        let groupKey = comp.name;
        if (groupByType) {
            const mesh = componentMeshes[comp.name];
            groupKey = mesh && mesh.userData.geometryFingerprint
                ? mesh.userData.geometryFingerprint
                : comp.name;
        }

        // Use custom display name if set; when grouped use type-based name, else full name
        const displayName = itemData && itemData.displayName
            ? itemData.displayName
            : groupByType
                ? formatComponentName(comp.name)
                : comp.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        // Get count for this component (default 1)
        const compCount = itemData && itemData.count !== undefined ? itemData.count : 1;

        if (!groups[groupKey]) {
            groups[groupKey] = {
                baseName: groupKey,
                displayName,
                representative: comp.name,
                count: 0
            };
        }
        groups[groupKey].count += compCount;
    });

    return Object.values(groups);
}

// ============================================================
// PNG Export
// ============================================================
async function exportPNG() {
    if (!manualConfig.steps.length) {
        showStatus('No steps to export', 'error');
        return;
    }

    showExportProgress(true, 'Preparing export...');
    const resolution = parseInt(document.getElementById('exportResolution').value) || 1200;
    const stepsPerRow = parseInt(document.getElementById('stepsPerRow').value) || 2;
    const stepsPerPage = parseInt(document.getElementById('stepsPerPage').value) || 4;

    try {
        // Render component overview images
        const uniqueComponents = getUniqueComponentsWithCounts();
        const componentImages = [];
        for (let i = 0; i < uniqueComponents.length; i++) {
            updateExportProgress(`Rendering component ${i + 1}/${uniqueComponents.length}...`,
                (i / (uniqueComponents.length + manualConfig.steps.length)) * 100);
            const comp = uniqueComponents[i];
            const useCAD = manualConfig.componentOverview && manualConfig.componentOverview.useCADOrientation;
            const img = await renderComponentAlone(comp.representative, 300, 300, useCAD);
            componentImages.push({ ...comp, image: img });
        }

        // Render step images
        const stepImages = [];
        for (let i = 0; i < manualConfig.steps.length; i++) {
            updateExportProgress(`Rendering step ${i + 1}/${manualConfig.steps.length}...`,
                ((uniqueComponents.length + i) / (uniqueComponents.length + manualConfig.steps.length)) * 100);
            const step = manualConfig.steps[i];
            const stepSize = step.size || 1;
            const imgW = stepSize === 2 ? resolution : resolution / stepsPerRow;
            const imgH = (resolution / stepsPerRow) * 0.75;
            const img = await renderStepToImage(step, imgW, imgH);
            stepImages.push({ ...step, image: img, renderWidth: imgW, renderHeight: imgH });
        }

        // Re-apply current step to scene
        if (currentStepIndex >= 0) {
            applyStepToScene(manualConfig.steps[currentStepIndex]);
        }

        updateExportProgress('Composing layout...', 90);

        // Compose final canvas
        await composeAndDownloadPNG(componentImages, stepImages, stepsPerRow, resolution);

        disposeExportRenderer();
        showExportProgress(false);
        showStatus('PNG exported successfully', 'success');
    } catch (e) {
        disposeExportRenderer();
        showExportProgress(false);
        showStatus('Export failed: ' + e.message, 'error');
    }
}

async function composeAndDownloadPNG(componentImages, stepImages, stepsPerRow, resolution) {
    const pageWidth = resolution;
    const margin = Math.round(resolution * 0.04);
    const contentWidth = pageWidth - margin * 2;

    // Calculate layout dimensions
    const titleHeight = Math.round(resolution * 0.07);
    const titlePadding = Math.round(resolution * 0.03);

    // Components section
    const compImgSize = Math.round(contentWidth / Math.max(componentImages.length, 4) * 0.7);
    const compLabelHeight = Math.round(resolution * 0.055);
    const compSectionHeight = componentImages.length > 0
        ? Math.round(resolution * 0.038) + compImgSize + compLabelHeight + margin
        : 0;

    // Steps section
    const stepColWidth = Math.round(contentWidth / stepsPerRow);
    const stepImgHeight = Math.round(stepColWidth * 0.75);
    const stepLabelHeight = Math.round(resolution * 0.035);
    const stepCellHeight = stepLabelHeight + stepImgHeight + Math.round(margin * 0.5);

    // Calculate total rows for steps
    let totalStepRows = 0;
    let colsUsed = 0;
    stepImages.forEach(step => {
        const size = step.size || 1;
        if (colsUsed + size > stepsPerRow) {
            totalStepRows++;
            colsUsed = 0;
        }
        colsUsed += size;
    });
    if (colsUsed > 0) totalStepRows++;

    const stepsSectionHeight = totalStepRows * stepCellHeight;
    const totalHeight = titleHeight + titlePadding + compSectionHeight + stepsSectionHeight + margin * 2;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = pageWidth;
    canvas.height = totalHeight;
    const ctx = canvas.getContext('2d');

    // White background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let y = margin;

    // Title
    ctx.fillStyle = '#000000';
    ctx.font = `bold ${Math.round(resolution * 0.04)}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(manualConfig.title, pageWidth / 2, y + titleHeight * 0.7);
    y += titleHeight + titlePadding;

    // Components section
    if (componentImages.length > 0) {
        ctx.font = `bold ${Math.round(resolution * 0.026)}px Arial, sans-serif`;
        ctx.textAlign = 'left';
        ctx.fillText('Components:', margin, y + Math.round(resolution * 0.026));
        y += Math.round(resolution * 0.038);

        const compSpacing = contentWidth / componentImages.length;
        for (let i = 0; i < componentImages.length; i++) {
            const comp = componentImages[i];
            if (!comp.image) continue;

            const x = margin + compSpacing * i + (compSpacing - compImgSize) / 2;

            await drawImageFromDataURL(ctx, comp.image, x, y, compImgSize, compImgSize);

            // Label
            ctx.fillStyle = '#000000';
            ctx.font = `${Math.round(resolution * 0.018)}px Arial, sans-serif`;
            ctx.textAlign = 'center';
            const labelX = x + compImgSize / 2;
            ctx.fillText(comp.displayName, labelX, y + compImgSize + Math.round(compLabelHeight * 0.45));
            ctx.fillText(`x${comp.count}`, labelX, y + compImgSize + Math.round(compLabelHeight * 0.85));
        }
        y += compImgSize + compLabelHeight + margin;
    }

    // Steps section
    colsUsed = 0;
    let rowY = y;
    for (let i = 0; i < stepImages.length; i++) {
        const step = stepImages[i];
        const size = step.size || 1;

        // Check if need new row
        if (colsUsed + size > stepsPerRow) {
            rowY += stepCellHeight;
            colsUsed = 0;
        }

        const cellX = margin + colsUsed * stepColWidth;
        const cellWidth = size * stepColWidth;

        // Step description
        ctx.fillStyle = '#000000';
        ctx.font = `bold ${Math.round(resolution * 0.018)}px Arial, sans-serif`;
        ctx.textAlign = 'left';
        ctx.fillText(step.description, cellX + 4, rowY + stepLabelHeight * 0.7);

        // Step image
        if (step.image) {
            const imgW = cellWidth - 8;
            const imgH = stepImgHeight - 4;
            await drawImageFromDataURL(ctx, step.image, cellX + 4, rowY + stepLabelHeight, imgW, imgH);
        }

        colsUsed += size;
    }

    // Download
    const link = document.createElement('a');
    link.download = `${manualConfig.title.replace(/\s+/g, '_')}_manual.png`;
    link.href = canvas.toDataURL('image/png');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function drawImageFromDataURL(ctx, dataURL, x, y, w, h) {
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, x, y, w, h);
            resolve();
        };
        img.onerror = resolve;
        img.src = dataURL;
    });
}

// ============================================================
// PDF Export
// ============================================================
async function exportPDF() {
    if (!manualConfig.steps.length) {
        showStatus('No steps to export', 'error');
        return;
    }

    showExportProgress(true, 'Preparing PDF export...');
    const resolution = parseInt(document.getElementById('exportResolution').value) || 1200;
    const stepsPerRow = parseInt(document.getElementById('stepsPerRow').value) || 2;

    try {
        // Render component images
        const uniqueComponents = getUniqueComponentsWithCounts();
        const componentPayload = [];
        for (let i = 0; i < uniqueComponents.length; i++) {
            updateExportProgress(`Rendering component ${i + 1}/${uniqueComponents.length}...`,
                (i / (uniqueComponents.length + manualConfig.steps.length)) * 100);
            const comp = uniqueComponents[i];
            const useCAD = manualConfig.componentOverview && manualConfig.componentOverview.useCADOrientation;
            const img = await renderComponentAlone(comp.representative, 300, 300, useCAD);
            if (img) {
                componentPayload.push({
                    name: comp.displayName,
                    count: comp.count,
                    image_b64: img.split(',')[1]
                });
            }
        }

        // Render step images
        const stepPayload = [];
        for (let i = 0; i < manualConfig.steps.length; i++) {
            updateExportProgress(`Rendering step ${i + 1}/${manualConfig.steps.length}...`,
                ((uniqueComponents.length + i) / (uniqueComponents.length + manualConfig.steps.length)) * 100);
            const step = manualConfig.steps[i];
            const imgW = (step.size === 2) ? resolution : resolution / stepsPerRow;
            const imgH = (resolution / stepsPerRow) * 0.75;
            const img = await renderStepToImage(step, imgW, imgH);
            stepPayload.push({
                description: step.description,
                image_b64: img.split(',')[1],
                size: step.size || 1
            });
        }

        // Re-apply current step
        if (currentStepIndex >= 0) {
            applyStepToScene(manualConfig.steps[currentStepIndex]);
        }

        updateExportProgress('Generating PDF...', 95);

        // Send to server
        const response = await fetch('/api/export-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: manualConfig.title,
                components: componentPayload,
                steps: stepPayload
            })
        });

        if (!response.ok) throw new Error('PDF generation failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${manualConfig.title.replace(/\s+/g, '_')}_manual.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        disposeExportRenderer();
        showExportProgress(false);
        showStatus('PDF exported successfully', 'success');
    } catch (e) {
        disposeExportRenderer();
        showExportProgress(false);
        showStatus('PDF export failed: ' + e.message, 'error');
    }
}

// ============================================================
// Save / Load Manual Config
// ============================================================
async function saveManualConfig() {
    if (!manualConfig.steps.length) {
        showStatus('No steps to save', 'error');
        return;
    }

    manualConfig.title = document.getElementById('manualTitle').value || 'Untitled';

    try {
        const response = await fetch('/api/save-manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: manualConfig.title,
                assemblyFile: manualConfig.assemblyFile,
                steps: manualConfig.steps,
                componentOverview: manualConfig.componentOverview,
                timestamp: new Date().toISOString()
            })
        });
        const result = await response.json();
        showStatus(`Manual saved as ${result.filename}`, 'success');
    } catch (e) {
        showStatus('Failed to save: ' + e.message, 'error');
    }
}

async function loadManualFromFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const text = await file.text();
        const config = JSON.parse(text);

        // Load the referenced assembly first
        if (config.assemblyFile) {
            document.getElementById('assemblySelect').value = config.assemblyFile;
            await loadAssembly();
        }

        // Restore manual config
        manualConfig.title = config.title || 'Untitled';
        manualConfig.steps = config.steps || [];
        manualConfig.componentOverview = config.componentOverview || {
            groupByType: true, useCADOrientation: false, items: {}
        };
        document.getElementById('manualTitle').value = manualConfig.title;

        // Migrate old formats
        manualConfig.steps.forEach(step => {
            Object.values(step.components).forEach(comp => {
                if (comp.zOffset !== undefined && !comp.offset) {
                    comp.offset = { x: 0, y: 0, z: comp.zOffset };
                    delete comp.zOffset;
                }
                if (!comp.rotation) {
                    comp.rotation = { x: 0, y: 0, z: 0 };
                }
            });
        });

        // Sync overview UI checkboxes
        const ov = manualConfig.componentOverview;
        document.getElementById('groupComponents').checked = ov.groupByType !== false;
        document.getElementById('cadOrientation').checked = !!ov.useCADOrientation;
        updateComponentsList();
        updateComponentEditor();

        if (manualConfig.steps.length > 0) {
            selectStep(0);
        }
        updateStepsList();
        showStatus('Manual loaded successfully', 'success');
    } catch (e) {
        showStatus('Failed to load manual: ' + e.message, 'error');
    }

    // Reset file input
    event.target.value = '';
}

// ============================================================
// Utility Functions
// ============================================================
function formatComponentName(name) {
    // Strip color suffixes and format nicely
    const colorSuffixes = ['_brown', '_orange', '_yellow', '_green', '_red', '_blue'];
    let display = name;
    for (const suffix of colorSuffixes) {
        if (display.endsWith(suffix)) {
            display = display.slice(0, -suffix.length);
            break;
        }
    }
    // Replace underscores with spaces and capitalize
    return display.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function showStatus(message, type) {
    const container = document.getElementById('statusMessages');
    const div = document.createElement('div');
    div.className = type || 'info';
    div.textContent = message;
    container.innerHTML = '';
    container.appendChild(div);

    // Auto-clear after 5 seconds
    setTimeout(() => {
        if (container.contains(div)) {
            container.removeChild(div);
        }
    }, 5000);
}

function showExportProgress(visible, message) {
    const overlay = document.getElementById('exportOverlay');
    overlay.style.display = visible ? 'flex' : 'none';
    if (message) {
        document.getElementById('exportStatus').textContent = message;
    }
    if (!visible) {
        document.getElementById('exportProgressFill').style.width = '0%';
    }
}

function updateExportProgress(message, percent) {
    document.getElementById('exportStatus').textContent = message;
    document.getElementById('exportProgressFill').style.width = `${percent}%`;
}

function updateLayoutPreview() {
    // Placeholder for future preview updates
}
