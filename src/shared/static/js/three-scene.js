/**
 * Shared Three.js scene utilities
 * Common setup for all 3D visualization apps
 */

/**
 * Initialize a Three.js scene with standard settings
 * @param {string} containerId - ID of the container element
 * @param {Object} options - Configuration options
 * @returns {Object} Scene components: scene, camera, renderer, controls
 */
function initThreeScene(containerId, options = {}) {
    const container = document.getElementById(containerId);
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(options.backgroundColor || 0x1a1a1a);

    // Camera - Orthographic for consistent dimensions when zooming
    const frustumSize = options.frustumSize || 0.5;
    const aspect = width / height;
    const camera = new THREE.OrthographicCamera(
        frustumSize * aspect / -2,
        frustumSize * aspect / 2,
        frustumSize / 2,
        frustumSize / -2,
        0.001,
        1000
    );
    camera.position.set(0.5, -0.5, 0.5);
    camera.up.set(0, 0, 1);  // Z-up convention
    camera.lookAt(0, 0, 0);
    camera.zoom = 1;
    camera.updateProjectionMatrix();

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
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

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    gridHelper.rotateX(Math.PI / 2);  // Horizontal in XY plane (Z-up)
    scene.add(gridHelper);

    // Custom coordinate frame (Z-up, Blender convention)
    const axesLength = options.axesLength || 0.5;
    const axesGeometry = new THREE.BufferGeometry();
    const axesMaterial = new THREE.LineBasicMaterial({ vertexColors: true });

    const axesVertices = new Float32Array([
        0, 0, 0,  axesLength, 0, 0,  // X axis (Red)
        0, 0, 0,  0, axesLength, 0,  // Y axis (Green)
        0, 0, 0,  0, 0, axesLength   // Z axis (Blue)
    ]);

    const axesColors = new Float32Array([
        1, 0, 0,  1, 0, 0,  // Red for X
        0, 1, 0,  0, 1, 0,  // Green for Y
        0, 0, 1,  0, 0, 1   // Blue for Z
    ]);

    axesGeometry.setAttribute('position', new THREE.BufferAttribute(axesVertices, 3));
    axesGeometry.setAttribute('color', new THREE.BufferAttribute(axesColors, 3));
    const axesHelper = new THREE.LineSegments(axesGeometry, axesMaterial);
    scene.add(axesHelper);

    // Handle window resize
    function handleResize() {
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        const newAspect = newWidth / newHeight;
        camera.left = frustumSize * newAspect / -2;
        camera.right = frustumSize * newAspect / 2;
        camera.top = frustumSize / 2;
        camera.bottom = frustumSize / -2;
        camera.updateProjectionMatrix();
        renderer.setSize(newWidth, newHeight);
    }
    window.addEventListener('resize', handleResize);

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    return {
        scene,
        camera,
        renderer,
        controls,
        gridHelper,
        axesHelper,
        handleResize
    };
}

/**
 * Fit camera to object bounds
 * @param {THREE.Camera} camera - The camera
 * @param {THREE.OrbitControls} controls - The orbit controls
 * @param {THREE.Object3D} object - The object to fit
 */
function fitCameraToObject(camera, controls, object) {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const distance = maxDim * 2;

    camera.position.set(center.x + distance, center.y - distance, center.z + distance);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

/**
 * Create a wireframe mesh from vertex and edge data
 * @param {Array} vertices - Array of [x, y, z] coordinates
 * @param {Array} edges - Array of [v1Index, v2Index] pairs
 * @param {number} color - Hex color for the wireframe
 * @returns {THREE.LineSegments} The wireframe mesh
 */
function createWireframeMesh(vertices, edges, color = 0x888888) {
    const positions = [];
    for (const [v1, v2] of edges) {
        const p1 = vertices[v1];
        const p2 = vertices[v2];
        positions.push(p1[0], p1[1], p1[2]);
        positions.push(p2[0], p2[1], p2[2]);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color });
    return new THREE.LineSegments(geometry, material);
}

/**
 * Create a mesh from vertex, normal, and face data
 * @param {Array} vertices - Flat array of vertex coordinates
 * @param {Array} normals - Flat array of normal vectors
 * @param {Array} faces - Flat array of face indices
 * @param {Object} options - Material options
 * @returns {THREE.Mesh} The mesh
 */
function createMeshFromData(vertices, normals, faces, options = {}) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    geometry.setIndex(faces);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
        color: options.color || 0x888888,
        side: THREE.DoubleSide,
        wireframe: options.wireframe || false
    });

    return new THREE.Mesh(geometry, material);
}
