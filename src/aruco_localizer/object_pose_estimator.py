import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import argparse
from pathlib import Path

def load_wireframe_data(json_file):
    """Load wireframe data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['vertices'], data['edges']

def load_aruco_annotations(json_file):
    """Load ArUco marker annotations from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['markers']

def get_available_models(data_dir):
    """Get list of available models from the data directory"""
    wireframe_dir = Path(data_dir) / "wireframe"
    aruco_dir = Path(data_dir) / "aruco"
    
    if not wireframe_dir.exists() or not aruco_dir.exists():
        return []
    
    # Get all wireframe files
    wireframe_files = list(wireframe_dir.glob("*_wireframe.json"))
    aruco_files = list(aruco_dir.glob("*_aruco.json"))
    
    # Extract model names (remove _wireframe.json and _aruco.json suffixes)
    wireframe_models = {f.stem.replace("_wireframe", "") for f in wireframe_files}
    aruco_models = {f.stem.replace("_aruco", "") for f in aruco_files}
    
    # Return intersection (models that have both wireframe and aruco files)
    available_models = wireframe_models.intersection(aruco_models)
    return sorted(list(available_models))

def select_model_interactive(available_models):
    """Interactive model selection"""
    if not available_models:
        print("No models found in data directory!")
        return None
    
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                selected_model = available_models[choice_num - 1]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def estimate_object_pose_from_marker(marker_pose, aruco_annotation):
    """
    Estimate the 6D pose of the object center from ArUco marker pose.
    
    Args:
        marker_pose: (tvec, rvec) - ArUco marker pose in camera frame
        aruco_annotation: ArUco annotation data containing marker-to-object transform
    
    Returns:
        object_pose: (object_tvec, object_rvec) - Object center pose in camera frame
    """
    # Get marker position and rotation
    marker_tvec, marker_rvec = marker_pose
    
    # Convert marker rotation vector to rotation matrix
    marker_rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
    
    # Get the marker's pose relative to CAD center from annotation
    marker_relative_pose = aruco_annotation['pose_relative_to_cad_center']
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Get marker position relative to object center (in object frame)
    marker_pos_in_object = np.array([
        marker_relative_pose['position']['x'],
        marker_relative_pose['position']['y'], 
        marker_relative_pose['position']['z']
    ])
    
    # Apply scaling and coordinate transformation
    marker_pos_in_object = coord_transform @ (marker_pos_in_object * 1.25)
    
    # Get marker orientation relative to object center
    marker_rot = marker_relative_pose['rotation']
    marker_rotation_in_object = euler_to_rotation_matrix(
        marker_rot['roll'], marker_rot['pitch'], marker_rot['yaw']
    )
    
    # Apply coordinate system transformation to the rotation matrix
    marker_rotation_in_object = coord_transform @ marker_rotation_in_object @ coord_transform.T
    
    # Calculate object center position in camera frame
    # The object center is at the origin of the object frame
    # We need to transform the origin (0,0,0) from object frame to camera frame
    object_origin_in_marker_frame = marker_rotation_in_object.T @ (-marker_pos_in_object)
    object_tvec = marker_tvec.flatten() + marker_rotation_matrix @ object_origin_in_marker_frame
    
    # Calculate object center orientation in camera frame
    # The object orientation is the marker orientation composed with the marker-to-object rotation
    object_rotation_matrix = marker_rotation_matrix @ marker_rotation_in_object.T
    
    # Convert back to rotation vector
    object_rvec, _ = cv2.Rodrigues(object_rotation_matrix)
    
    return object_tvec, object_rvec

def transform_mesh_to_camera_frame(vertices, object_pose):
    """Transform mesh vertices from object center frame to camera frame"""
    object_tvec, object_rvec = object_pose
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(object_rvec)
    
    # Coordinate system transformation matrix
    coord_transform = np.array([
        [-1,  0,  0],  # X-axis: flip (3D graphics X-right → OpenCV X-left)
        [0,   1,  0],  # Y-axis: unchanged (both systems use Y-up)
        [0,   0, -1]   # Z-axis: flip (3D graphics Z-forward → OpenCV Z-backward)
    ])
    
    # Transform vertices from object center frame to camera frame
    transformed_vertices = []
    for vertex in vertices:
        # Apply coordinate system transformation and scaling
        vertex_transformed = coord_transform @ (np.array(vertex) * 1.25)
        
        # Transform from object frame to camera frame
        vertex_cam = rotation_matrix @ vertex_transformed + object_tvec
        transformed_vertices.append(vertex_cam)
    
    return np.array(transformed_vertices)

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
    # Convert to radians if needed (assuming they're already in radians from JSON)
    r, p, y = roll, pitch, yaw
    
    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    
    Ry = np.array([[np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    
    # Combine rotations (order: Rz * Ry * Rx)
    return Rz @ Ry @ Rx

def project_vertices_to_image(vertices, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if len(vertices) == 0:
        return np.array([])
    
    # Project points to image plane
    projected_points, _ = cv2.projectPoints(
        vertices.astype(np.float32), 
        np.zeros((3, 1)),  # No rotation (already in camera frame)
        np.zeros((3, 1)),  # No translation (already in camera frame)
        camera_matrix, 
        dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(np.int32)

def draw_wireframe(frame, projected_vertices, edges, color=(0, 255, 0), thickness=2):
    """Draw wireframe on the image"""
    if len(projected_vertices) == 0:
        return
    
    # Filter out vertices that are outside the image bounds
    height, width = frame.shape[:2]
    valid_vertices = []
    valid_indices = []
    
    for i, vertex in enumerate(projected_vertices):
        x, y = vertex
        if 0 <= x < width and 0 <= y < height:
            valid_vertices.append(vertex)
            valid_indices.append(i)
    
    if len(valid_vertices) == 0:
        return
    
    # Create mapping from original indices to valid indices
    index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
    
    # Draw edges
    for edge in edges:
        if len(edge) >= 2:
            start_idx, end_idx = edge[0], edge[1]
            if start_idx in index_map and end_idx in index_map:
                start_point = tuple(valid_vertices[index_map[start_idx]])
                end_point = tuple(valid_vertices[index_map[end_idx]])
                cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw vertices as small circles
    for vertex in valid_vertices:
        cv2.circle(frame, tuple(vertex), 3, (255, 0, 0), -1)

def detect_object_pose_with_mesh_overlay(model_name=None, data_dir=None):
    """Detect ArUco marker and overlay mesh wireframe"""
    
    # Set default data directory if not provided
    if data_dir is None:
        # Assume we're running from aruco_localizer directory, go up to find data
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent.parent / "data"
    
    data_dir = Path(data_dir)
    
    # Get available models
    available_models = get_available_models(data_dir)
    
    if not available_models:
        print(f"No models found in data directory: {data_dir}")
        return
    
    # Select model
    if model_name is None:
        model_name = select_model_interactive(available_models)
        if model_name is None:
            print("No model selected. Exiting.")
            return
    elif model_name not in available_models:
        print(f"Model '{model_name}' not found in available models: {available_models}")
        return
    
    # Construct file paths
    wireframe_file = data_dir / "wireframe" / f"{model_name}_wireframe.json"
    aruco_annotations_file = data_dir / "aruco" / f"{model_name}_aruco.json"
    
    print(f"Using model: {model_name}")
    print(f"Wireframe file: {wireframe_file}")
    print(f"ArUco annotations file: {aruco_annotations_file}")
    
    # Load wireframe data
    try:
        vertices, edges = load_wireframe_data(wireframe_file)
        print(f"Loaded wireframe: {len(vertices)} vertices, {len(edges)} edges")
    except Exception as e:
        print(f"Error loading wireframe data: {e}")
        return
    
    # Load ArUco annotations
    try:
        aruco_annotations = load_aruco_annotations(aruco_annotations_file)
        print(f"Loaded {len(aruco_annotations)} ArUco annotations")
    except Exception as e:
        print(f"Error loading ArUco annotations: {e}")
        return
    
    # Create a dictionary mapping marker IDs to their annotations
    marker_annotations = {}
    for annotation in aruco_annotations:
        marker_id = annotation['aruco_id']
        marker_annotations[marker_id] = annotation
        print(f"Loaded annotation for marker ID {marker_id}: size={annotation['size']}m, border={annotation['border_width']}m, face={annotation['face_type']}")
    
    print(f"Total markers available: {len(marker_annotations)}")
    
    # Camera parameters
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # ArUco parameters - we'll calculate marker sizes dynamically for each detected marker
    target_ids = list(marker_annotations.keys())  # All available marker IDs
    
    print(f"Looking for markers: {target_ids}")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Open camera
    cap = cv2.VideoCapture(8)  # Using camera ID 8
    if not cap.isOpened():
        print("Failed to open camera 8")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Object Pose Estimation Started")
    print(f"Looking for markers {target_ids} to estimate object pose")
    print("Press 'q' to quit, 's' to toggle mesh display")
    print("=" * 50)
    
    show_mesh = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Draw all detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Check for any of our target markers
            detected_targets = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in target_ids:
                    detected_targets.append((i, marker_id))
            
            if detected_targets:
                # First pass: collect all successful pose estimations
                successful_detections = []
                for target_idx, marker_id in detected_targets:
                    target_corners = corners[target_idx]
                    marker_annotation = marker_annotations[marker_id]
                    
                    # Calculate marker size including border
                    base_marker_size = marker_annotation['size']
                    border_percentage = marker_annotation['border_width']
                    marker_size = base_marker_size * (1 + border_percentage)
                    
                    # Estimate pose
                    object_points = np.array([
                        [-marker_size/2,  marker_size/2, 0],
                        [ marker_size/2,  marker_size/2, 0],
                        [ marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(object_points, target_corners[0], camera_matrix, dist_coeffs)
                    
                    if success:
                        position = tvec.flatten()
                        distance = np.linalg.norm(position)
                        successful_detections.append({
                            'target_idx': target_idx,
                            'marker_id': marker_id,
                            'marker_annotation': marker_annotation,
                            'target_corners': target_corners,
                            'marker_size': marker_size,
                            'rvec': rvec,
                            'tvec': tvec,
                            'position': position,
                            'distance': distance
                        })
                
                # Find the most confident marker (closest to camera)
                best_marker = None
                if successful_detections:
                    best_marker = min(successful_detections, key=lambda x: x['distance'])
                
                # Process all detected markers for display, but only show wireframe for the best one
                for i, detection in enumerate(successful_detections):
                    target_idx = detection['target_idx']
                    marker_id = detection['marker_id']
                    marker_annotation = detection['marker_annotation']
                    target_corners = detection['target_corners']
                    marker_size = detection['marker_size']
                    rvec = detection['rvec']
                    tvec = detection['tvec']
                    position = detection['position']
                    distance = detection['distance']
                    face_type = marker_annotation['face_type']
                    
                    # Draw coordinate axes for all markers
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
                    # Only show wireframe for the most confident (closest) marker
                    is_best_marker = (detection == best_marker)
                    if show_mesh and is_best_marker:
                        # Estimate object pose from marker pose
                        object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                        
                        # Transform mesh vertices to camera frame using object pose
                        transformed_vertices = transform_mesh_to_camera_frame(vertices, (object_tvec, object_rvec))
                        
                        # Project vertices to image coordinates
                        projected_vertices = project_vertices_to_image(transformed_vertices, camera_matrix, dist_coeffs)
                        
                        # Draw wireframe
                        if len(projected_vertices) > 0:
                            draw_wireframe(frame, projected_vertices, edges, color=(0, 255, 0), thickness=2)
                        
                        # Draw object center coordinate axes
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, object_rvec, object_tvec, marker_size * 1.5)
                    
                    # Display information for this marker
                    # Position text based on marker index to avoid overlap
                    y_offset = 30 + i * 150  # Increased spacing for object pose info
                    
                    # Use different colors for best vs other markers
                    text_color = (0, 255, 0) if is_best_marker else (0, 255, 255)
                    marker_status = " (BEST)" if is_best_marker else ""
                    
                    cv2.putText(frame, f"Marker ID: {marker_id} ({face_type}){marker_status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    cv2.putText(frame, f"Marker Pos: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                               (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(frame, f"Marker Dist: {distance:.3f}m", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Show object pose for best marker
                    if is_best_marker:
                        object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                        object_distance = np.linalg.norm(object_tvec)
                        cv2.putText(frame, f"Object Pos: ({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f})", 
                                   (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, f"Object Dist: {object_distance:.3f}m", (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Print to console for the best marker
                        print(f"\rBest Marker {marker_id} ({face_type}): Marker=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) | "
                              f"Object=({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f}) | "
                              f"Dist: {object_distance:.3f}m | Mesh: {'ON' if show_mesh else 'OFF'}", end="", flush=True)
                
                # Display mesh status
                cv2.putText(frame, f"Mesh: {'ON' if show_mesh else 'OFF'}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"Looking for markers {target_ids}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show controls
        cv2.putText(frame, "Press 'q' to quit, 's' to toggle mesh", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Object Pose Estimation", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_mesh = not show_mesh
            print(f"\nMesh display: {'ON' if show_mesh else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nObject pose estimation stopped.")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Object Pose Estimator - Estimate 6D object pose from ArUco markers")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Model name to use (e.g., 'fork_orange_scaled70'). If not provided, interactive selection will be used.")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                       help="Path to data directory containing wireframe and aruco subdirectories")
    parser.add_argument("--list-models", "-l", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    
    # Get available models
    available_models = get_available_models(data_dir)
    
    if args.list_models:
        print(f"Available models in {data_dir}:")
        if available_models:
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
        else:
            print("  No models found!")
        return
    
    if not available_models:
        print(f"No models found in data directory: {data_dir}")
        return
    
    # Run the object pose estimation
    detect_object_pose_with_mesh_overlay(model_name=args.model, data_dir=data_dir)

if __name__ == "__main__":
    main()
