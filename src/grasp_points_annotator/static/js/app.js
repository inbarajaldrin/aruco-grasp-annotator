/**
 * Grasp Points Annotation App
 * Main JavaScript application file
 */

// Application state
let currentObject = null;
let currentMarker = null;
let step1Result = null;
let step2Result = null;

// Initialize on page load
window.addEventListener('load', async function() {
    await loadObjects();
});

/**
 * Load available objects from the API
 */
async function loadObjects() {
    const select = document.getElementById('objectSelect');
    select.innerHTML = '<option value="">Loading objects...</option>';

    try {
        const response = await fetch('/api/objects');

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.objects || data.objects.length === 0) {
            select.innerHTML = '<option value="">No objects found</option>';
            showMessage('No objects found. Make sure data/models/ and data/aruco/ directories contain matching files.', 'warning');
            return;
        }

        select.innerHTML = '<option value="">Select an object...</option>';

        data.objects.forEach(function(obj) {
            const option = document.createElement('option');
            option.value = obj;
            option.textContent = obj;
            select.appendChild(option);
        });

        // Set up event listener
        const newSelect = select.cloneNode(true);
        select.parentNode.replaceChild(newSelect, select);
        document.getElementById('objectSelect').addEventListener('change', async function(e) {
            currentObject = e.target.value;
            if (currentObject) {
                await loadMarkers(currentObject);
            } else {
                document.getElementById('markerSelect').innerHTML = '<option value="">Select object first</option>';
                document.getElementById('markerSelect').disabled = true;
            }
        });

        showMessage(`Loaded ${data.objects.length} object(s)`, 'success');
    } catch (error) {
        select.innerHTML = '<option value="">Error loading objects</option>';
        showMessage('Error loading objects: ' + error.message, 'error');
        console.error('Error loading objects:', error);
    }
}

/**
 * Load markers for a specific object
 */
async function loadMarkers(objectName) {
    const select = document.getElementById('markerSelect');
    select.innerHTML = '<option value="">Loading markers...</option>';
    select.disabled = true;

    try {
        const response = await fetch(`/api/markers/${encodeURIComponent(objectName)}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.markers || data.markers.length === 0) {
            select.innerHTML = '<option value="">No markers found</option>';
            select.disabled = true;
            showMessage(`No markers found for ${objectName}`, 'warning');
            return;
        }

        select.innerHTML = '<option value="">Select a marker...</option>';
        select.disabled = false;

        data.markers.forEach(function(markerId) {
            const option = document.createElement('option');
            option.value = markerId;
            option.textContent = `Marker ${markerId}`;
            select.appendChild(option);
        });

        // Set up event listener
        const newSelect = select.cloneNode(true);
        select.parentNode.replaceChild(newSelect, select);
        document.getElementById('markerSelect').addEventListener('change', function(e) {
            currentMarker = e.target.value ? parseInt(e.target.value) : null;
        });
    } catch (error) {
        select.innerHTML = '<option value="">Error loading markers</option>';
        select.disabled = true;
        showMessage('Error loading markers: ' + error.message, 'error');
        console.error('Error loading markers:', error);
    }
}

/**
 * Run Step 1: CAD to Grasp Points
 */
async function runStep1() {
    if (!currentObject || !currentMarker) {
        showMessage('Please select an object and marker first', 'warning');
        return;
    }

    const btn = document.getElementById('runStep1');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const statusDiv = document.getElementById('step1-status');
    statusDiv.innerHTML = '<div class="status-message info">Running Step 1: Rendering and detecting grasp points...</div>';

    try {
        const cameraDistance = parseFloat(document.getElementById('cameraDistance').value);
        const minAreaThreshold = parseInt(document.getElementById('minAreaThreshold').value);
        const useMtlColor = document.getElementById('useMtlColor').checked;

        const response = await fetch('/api/pipeline/step1', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                object_name: currentObject,
                marker_id: currentMarker,
                camera_distance: cameraDistance,
                min_area_threshold: minAreaThreshold,
                use_mtl_color: useMtlColor
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Step 1 failed');
        }

        step1Result = await response.json();

        statusDiv.innerHTML = `
            <div class="status-message success">
                Step 1 completed!<br>
                Detected ${step1Result.points_2d_count} 2D regions<br>
                Mapped ${step1Result.points_3d_count} 3D grasp points
            </div>
        `;

        displayStep1Results(step1Result);

        document.getElementById('applyFilter').disabled = false;
        document.getElementById('runStep2').disabled = false;

        showMessage('Step 1 completed successfully!', 'success');
    } catch (error) {
        statusDiv.innerHTML = `<div class="status-message error">Error: ${error.message}</div>`;
        showMessage('Step 1 failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Step 1';
    }
}

/**
 * Apply grasp point filter
 */
async function applyFilter() {
    if (!step1Result) {
        showMessage('Please run Step 1 first', 'warning');
        return;
    }

    const btn = document.getElementById('applyFilter');
    btn.disabled = true;
    btn.textContent = 'Filtering...';

    const statusDiv = document.getElementById('filter-status');
    statusDiv.innerHTML = '<div class="status-message info">Applying grasp filter...</div>';

    try {
        const gripperMaxWidth = parseFloat(document.getElementById('gripperMaxWidth').value);
        const gripperHalfOpen = parseFloat(document.getElementById('gripperHalfOpen').value);
        const gripperTipThickness = parseFloat(document.getElementById('gripperTipThickness').value);
        const maxGapPx = parseInt(document.getElementById('maxGapPx').value);
        const symmetryTolerance = parseFloat(document.getElementById('symmetryTolerance').value);
        const checkXAxis = document.getElementById('checkXAxis').checked;
        const checkYAxis = document.getElementById('checkYAxis').checked;

        const response = await fetch('/api/filter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                object_name: currentObject,
                marker_id: currentMarker,
                gripper_max_width_mm: gripperMaxWidth,
                gripper_half_open_width_mm: gripperHalfOpen,
                gripper_tip_thickness_mm: gripperTipThickness,
                max_gap_px: maxGapPx,
                symmetry_tolerance_mm: symmetryTolerance,
                check_x_axis: checkXAxis,
                check_y_axis: checkYAxis
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Filter failed');
        }

        const filterResult = await response.json();

        statusDiv.innerHTML = `
            <div class="status-message success">
                Filter applied!<br>
                Original: ${filterResult.total_original} grasp points<br>
                Filtered: ${filterResult.total_filtered} grasp points<br>
                Removed: ${filterResult.total_original - filterResult.total_filtered} grasp points
            </div>
        `;

        displayFilterResults(filterResult);

        showMessage('Filter applied successfully!', 'success');
    } catch (error) {
        statusDiv.innerHTML = `<div class="status-message error">Error: ${error.message}</div>`;
        showMessage('Filter failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Apply Filter';
    }
}

/**
 * Run Step 2: Transform to All Markers
 */
async function runStep2() {
    if (!step1Result) {
        showMessage('Please run Step 1 first', 'warning');
        return;
    }

    const btn = document.getElementById('runStep2');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const statusDiv = document.getElementById('step2-status');
    statusDiv.innerHTML = '<div class="status-message info">Running Step 2: Transforming to all markers...</div>';

    try {
        const objectThickness = document.getElementById('objectThickness').value;
        const payload = {
            object_name: currentObject,
            source_marker_id: currentMarker
        };
        if (objectThickness) {
            payload.object_thickness = parseFloat(objectThickness);
        }

        const response = await fetch('/api/pipeline/step2', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Step 2 failed');
        }

        step2Result = await response.json();

        statusDiv.innerHTML = `
            <div class="status-message success">
                Step 2 completed!<br>
                Total grasp points: ${step2Result.total_grasp_points}<br>
                Total markers: ${step2Result.total_markers}
            </div>
        `;

        displayStep2Results(step2Result);

        document.getElementById('downloadBtn').disabled = false;

        showMessage('Step 2 completed successfully!', 'success');
    } catch (error) {
        statusDiv.innerHTML = `<div class="status-message error">Error: ${error.message}</div>`;
        showMessage('Step 2 failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Step 2';
    }
}

/**
 * Display Step 1 results in the viewer
 */
function displayStep1Results(result) {
    const viewer = document.getElementById('viewer');
    let html = `
        <h3 class="viewer-title">Step 1 Results</h3>
        <div class="images-row">
            <div class="image-card">
                <h4 class="viewer-subtitle">Rendered Top-Down View</h4>
                <img src="${result.rendered_image}" class="image-viewer" alt="Rendered image">
            </div>
    `;

    if (result.visualization_image) {
        html += `
            <div class="image-card">
                <h4 class="viewer-subtitle">Grasp Points Visualization</h4>
                <img src="${result.visualization_image}" class="image-viewer" alt="Grasp points">
            </div>
        `;
    }

    html += '</div>';
    viewer.innerHTML = html;
}

/**
 * Display filter results in the viewer
 */
function displayFilterResults(result) {
    const viewer = document.getElementById('viewer');
    const initialViz = step1Result && step1Result.visualization_image ? step1Result.visualization_image : null;

    let html = `
        <div class="filter-results-container">
            <h3 class="viewer-title" style="text-align: center;">Filter Results</h3>
            <div class="images-row" style="margin-bottom: 30px;">
    `;

    if (initialViz) {
        html += `
            <div class="image-card">
                <h4 class="viewer-subtitle">Initial Grasp Points</h4>
                <img src="${initialViz}" class="image-viewer" alt="Initial grasp points">
                <p class="image-caption">${result.total_original} grasp points detected</p>
            </div>
        `;
    }

    if (result.filtered_visualization) {
        html += `
            <div class="image-card">
                <h4 class="viewer-subtitle">Filtered Grasp Points</h4>
                <img src="${result.filtered_visualization}" class="image-viewer" alt="Filtered grasp points">
                <p class="image-caption">${result.total_filtered} valid grasp points (${result.total_original - result.total_filtered} removed)</p>
            </div>
        `;
    }

    html += `
            </div>
            <div class="viewer-info" style="padding: 0 20px;">
                <p><strong>Object:</strong> ${result.object_name}</p>
                <p><strong>Marker:</strong> ${result.marker_id}</p>
                <p><strong>Original Grasp Points:</strong> ${result.total_original}</p>
                <p><strong>Filtered Grasp Points:</strong> ${result.total_filtered}</p>
                <p><strong>Removed:</strong> ${result.total_original - result.total_filtered}</p>
            </div>
            <div class="viewer-info" style="padding: 0 20px;">
                <h4>Filter Parameters:</h4>
                <div style="overflow-x: auto;">
                    <table class="results-table" style="background: white; margin-top: 10px;">
                        <tbody style="color: #2c3e50;">
                            <tr>
                                <td><strong>Gripper Max Width</strong></td>
                                <td>${result.filter_params.gripper_max_width_mm} mm</td>
                            </tr>
                            <tr>
                                <td><strong>Gripper Half-Open Width</strong></td>
                                <td>${result.filter_params.gripper_half_open_width_mm} mm</td>
                            </tr>
                            <tr>
                                <td><strong>Gripper Tip Thickness</strong></td>
                                <td>${result.filter_params.gripper_tip_thickness_mm} mm</td>
                            </tr>
                            <tr>
                                <td><strong>Max Gap</strong></td>
                                <td>${result.filter_params.max_gap_px} px</td>
                            </tr>
                            <tr>
                                <td><strong>Symmetry Tolerance</strong></td>
                                <td>${result.filter_params.symmetry_tolerance_mm} mm</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="viewer-info" style="margin-top: 20px; padding: 0 20px; margin-bottom: 30px;">
                <h4>Grasp Point Validity:</h4>
                <div style="overflow-x: auto;">
                    <table class="results-table" style="background: white; margin-top: 10px;">
                        <thead>
                            <tr>
                                <th>Grasp ID</th>
                                <th>X-axis</th>
                                <th>Y-axis</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody style="color: #2c3e50;">
                            ${result.filter_results.map(function(fr) {
                                const bgClass = fr.is_valid ? 'validity-valid' : 'validity-invalid';
                                const xAxisText = fr.valid_x.length > 0 ? fr.valid_x.join(', ') : 'BLOCKED';
                                const yAxisText = fr.valid_y.length > 0 ? fr.valid_y.join(', ') : 'BLOCKED';
                                const statusText = fr.is_valid ? 'Valid' : 'Removed';
                                return `
                                    <tr class="${bgClass}">
                                        <td>${fr.grasp_id}</td>
                                        <td>${xAxisText}</td>
                                        <td>${yAxisText}</td>
                                        <td>${statusText}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    viewer.innerHTML = html;
}

/**
 * Display Step 2 results in the viewer
 */
function displayStep2Results(result) {
    const viewer = document.getElementById('viewer');

    let html = '<h3 class="viewer-title">Step 2 Results</h3>';

    if (result.renumbered_visualization) {
        html += `
            <div class="image-card" style="margin-bottom: 30px;">
                <h4 class="viewer-subtitle">Renumbered Grasp Points</h4>
                <img src="${result.renumbered_visualization}" class="image-viewer" alt="Renumbered grasp points">
                <p class="image-caption">Grasp points renumbered sequentially (1, 2, 3...) matching the output JSON</p>
            </div>
        `;
    }

    html += `
        <div class="viewer-info">
            <p><strong>Object:</strong> ${result.object_name}</p>
            <p><strong>Source Marker:</strong> ${result.source_marker_id}</p>
            <p><strong>Total Grasp Points:</strong> ${result.total_grasp_points}</p>
            <p><strong>Total Markers:</strong> ${result.total_markers}</p>
            <p><strong>Output File:</strong> ${result.output_file}</p>
        </div>
        <div class="viewer-info">
            <h4>Grasp Points (relative to CAD center):</h4>
            <table class="results-table" style="background: white; margin-top: 10px;">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>X (m)</th>
                        <th>Y (m)</th>
                        <th>Z (m)</th>
                    </tr>
                </thead>
                <tbody style="color: #2c3e50;">
                    ${result.grasp_points.map(function(gp) {
                        return `
                            <tr>
                                <td>${gp.id}</td>
                                <td>${gp.position.x.toFixed(4)}</td>
                                <td>${gp.position.y.toFixed(4)}</td>
                                <td>${gp.position.z.toFixed(4)}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;

    viewer.innerHTML = html;
}

/**
 * Download results
 */
function downloadResults() {
    if (!currentObject) {
        showMessage('No results to download', 'warning');
        return;
    }

    window.location.href = `/api/download/${currentObject}`;
}

/**
 * Show a status message
 */
function showMessage(message, type) {
    const container = document.getElementById('statusMessages');
    const div = document.createElement('div');
    div.className = `status-message ${type}`;
    div.textContent = message;
    container.appendChild(div);

    setTimeout(function() {
        if (div.parentNode) {
            div.parentNode.removeChild(div);
        }
    }, 5000);
}
