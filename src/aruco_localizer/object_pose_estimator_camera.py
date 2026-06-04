import cv2
import cv2.aruco as aruco
import numpy as np
import json
import argparse
from pathlib import Path

# Shared utilities imported from core (single source of truth); no local duplicates.
from core.kalman_filter import (
    QuaternionKalman,
    MAX_MOVEMENT_THRESHOLD,
    HOLD_REQUIRED_FRAMES,
    GHOST_TRACKING_FRAMES,
    BLEND_FACTOR,
    PROCESS_NOISE_POSITION,
    PROCESS_NOISE_QUATERNION,
    PROCESS_NOISE_VELOCITY,
    MEASUREMENT_NOISE_POSITION,
    MEASUREMENT_NOISE_QUATERNION,
)
from core.pose_math import (
    rvec_to_quat,
    quat_to_rvec,
    slerp_quat,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    estimate_object_pose_from_marker,
    pose_to_world,
)
from core.model_io import (
    load_wireframe_data,
    load_aruco_annotations,
    get_available_models,
    select_model_interactive,
)
from core.mesh_ops import (
    transform_mesh_to_camera_frame,
    project_vertices_to_image,
    draw_wireframe,
)

# Camera parameters
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV = 69.4  # degrees
CAMERA_VFOV = 42.5  # degrees

# Camera orientation (world <- camera). Assumes the camera is facing the marker from
# above (top-down) so we can lift camera-frame poses into a nominal world frame.
CAMERA_QUAT_WORLD = np.array([0.0, 1.0, 0.0, 0.0])  # [x, y, z, w]

# ArUco parameters
MARKER_SIZE = 0.021  # meters - adjust based on your markers
ARUCO_DICTIONARY = aruco.DICT_4X4_50


def estimate_pose_with_kalman(frame, corners, ids, camera_matrix, dist_coeffs, marker_size,
                             kalman_filters, marker_stabilities, last_seen_frames, current_frame):
    """Estimate pose with Kalman filtering and stability checking"""
    
    if corners and ids:
        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)
            
            # Initialize tracking state if this is a new marker
            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = QuaternionKalman()
                marker_stabilities[marker_id] = {
                    "last_tvec": None,
                    "last_frame": -1,
                    "confirmed": False,
                    "hold_counter": 0
                }
                last_seen_frames[marker_id] = 0
            
            kalman = kalman_filters[marker_id]
            stability = marker_stabilities[marker_id]
            
            # Prepare points for solvePnP
            image_points = corner[0].reshape(-1, 2)
            half_size = marker_size / 2
            object_points = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ], dtype=np.float32)
            
            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                tvec_flat = tvec.flatten()
                distance = np.linalg.norm(tvec_flat - stability["last_tvec"]) if stability["last_tvec"] is not None else 0
                movement_ok = distance < MAX_MOVEMENT_THRESHOLD
                
                if movement_ok:
                    stability["hold_counter"] += 1
                else:
                    stability["hold_counter"] = 0
                
                stability["last_tvec"] = tvec_flat
                stability["last_frame"] = current_frame
                
                if stability["hold_counter"] >= HOLD_REQUIRED_FRAMES:
                    stability["confirmed"] = True
                    
                    # Apply Kalman filtering with blending
                    measured_quat = rvec_to_quat(rvec)
                    pred_tvec, pred_rvec = kalman.predict()
                    pred_quat = rvec_to_quat(pred_rvec)
                    
                    # Blend measurements with predictions
                    blended_quat = slerp_quat(pred_quat, measured_quat, blend=BLEND_FACTOR)
                    blended_rvec = quat_to_rvec(blended_quat)
                    blended_tvec = BLEND_FACTOR * tvec_flat + (1 - BLEND_FACTOR) * pred_tvec
                    
                    # Update Kalman filter
                    kalman.correct(blended_tvec, blended_rvec)
                    last_seen_frames[marker_id] = current_frame
                    
                    return blended_tvec, blended_rvec, marker_id, True
                else:
                    return tvec_flat, rvec.flatten(), marker_id, False
    
    # Handle ghost tracking for confirmed markers
    for marker_id, kalman in kalman_filters.items():
        stability = marker_stabilities[marker_id]
        last_seen = last_seen_frames[marker_id]
        
        if not stability["confirmed"]:
            continue
        
        if current_frame - last_seen < GHOST_TRACKING_FRAMES:
            pred_tvec, pred_rvec = kalman.predict()
            return pred_tvec, pred_rvec, marker_id, True
        else:
            stability["confirmed"] = False
    
    return None, None, None, False

def detect_object_pose_with_kalman_filtering(model_name=None, data_dir=None, camera_id=None):
    """Detect ArUco marker and overlay mesh wireframe with Kalman filtering"""
    
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
        (
            aruco_annotations,
            base_marker_size,
            border_width_percent,
            aruco_dict_name,
        ) = load_aruco_annotations(aruco_annotations_file)
        print(f"Loaded {len(aruco_annotations)} ArUco annotations")
        print(f"Marker size: {base_marker_size}m, border width: {border_width_percent}")
        print(f"ArUco dictionary: {aruco_dict_name}")
    except Exception as e:
        print(f"Error loading ArUco annotations: {e}")
        return
    
    # Create a dictionary mapping marker IDs to their annotations
    marker_annotations = {}
    for annotation in aruco_annotations:
        marker_id = annotation['aruco_id']
        marker_annotations[marker_id] = annotation
        print(f"Loaded annotation for marker ID {marker_id}: face={annotation['face_type']}")
    
    print(f"Total markers available: {len(marker_annotations)}")
    
    # Calculate camera matrix from field of view
    fx = CAMERA_WIDTH / (2 * np.tan(np.deg2rad(CAMERA_HFOV / 2)))
    fy = CAMERA_HEIGHT / (2 * np.tan(np.deg2rad(CAMERA_VFOV / 2)))
    camera_matrix = np.array([[fx, 0, CAMERA_WIDTH / 2],
                              [0, fy, CAMERA_HEIGHT / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    print(f"Camera matrix: fx={fx:.1f}, fy={fy:.1f}")
    
    # ArUco parameters
    target_ids = list(marker_annotations.keys())
    print(f"Looking for markers: {target_ids}")
    # Select dictionary based on the JSON metadata (defaults to 4x4 if missing/invalid)
    try:
        dictionary_id = getattr(aruco, aruco_dict_name)
    except AttributeError:
        print(f"Warning: Unknown dictionary '{aruco_dict_name}', falling back to DICT_4X4_50")
        dictionary_id = aruco.DICT_4X4_50
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Select camera if not provided
    if camera_id is None:
        try:
            from camera_streamer import detect_available_cameras, select_camera
            available_cameras = detect_available_cameras()
            if not available_cameras:
                print("No cameras detected! Using default camera ID 0.")
                camera_id = 0
            else:
                camera_id = select_camera(available_cameras)
                if camera_id is None:
                    print("No camera selected. Using default camera ID 0.")
                    camera_id = 0
        except ImportError:
            print("Warning: camera_streamer module not found. Using default camera ID 0.")
            camera_id = 0
        except Exception as e:
            print(f"Warning: Error during camera detection: {e}. Using default camera ID 0.")
            camera_id = 0
    
    # Open camera with error handling
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_id}")
            return
        
        print(f"Using camera {camera_id}")
    except Exception as e:
        print(f"Error opening camera {camera_id}: {e}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("Object Pose Estimation with Kalman Filtering Started")
    print(f"Looking for markers {target_ids} to estimate object pose")
    print("Press 'q' to quit, 's' to toggle mesh display")
    print("=" * 60)
    print(f"Kalman Filter Settings:")
    print(f"  Max Movement: {MAX_MOVEMENT_THRESHOLD}m")
    print(f"  Hold Required: {HOLD_REQUIRED_FRAMES} frames")
    print(f"  Ghost Tracking: {GHOST_TRACKING_FRAMES} frames")
    print(f"  Blend Factor: {BLEND_FACTOR}")
    print("=" * 60)
    
    show_mesh = True
    current_frame = 0
    
    # Kalman filter tracking
    kalman_filters = {}
    marker_stabilities = {}
    last_seen_frames = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        current_frame += 1
        
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
                # First pass: collect all successful pose estimations with Kalman filtering
                successful_detections = []
                for target_idx, marker_id in detected_targets:
                    target_corners = corners[target_idx]
                    marker_annotation = marker_annotations[marker_id]
                    
                    # Calculate actual ArUco pattern size (border is INSIDE the total marker size)
                    # Match external aruco_camera_localizer: TOTAL_MARKER_SIZE = MARKER_SIZE - 2 * BORDER_WIDTH
                    border_width = base_marker_size * border_width_percent
                    total_marker_size = base_marker_size - 2 * border_width  # Actual ArUco pattern size
                    
                    # Estimate pose with Kalman filtering
                    tvec, rvec, filtered_marker_id, is_confirmed = estimate_pose_with_kalman(
                        frame, [target_corners], [marker_id], camera_matrix, dist_coeffs, 
                        total_marker_size, kalman_filters, marker_stabilities, last_seen_frames, current_frame
                    )
                    
                    if tvec is not None and rvec is not None:
                        position = tvec.flatten()
                        distance = np.linalg.norm(position)
                        successful_detections.append({
                            'target_idx': target_idx,
                            'marker_id': marker_id,
                            'marker_annotation': marker_annotation,
                            'target_corners': target_corners,
                            'marker_size': total_marker_size,
                            'rvec': rvec,
                            'tvec': tvec,
                            'position': position,
                            'distance': distance,
                            'is_confirmed': is_confirmed
                        })
                
                # Find the most confident marker (closest to camera) among confirmed detections
                best_marker = None
                confirmed_detections = [d for d in successful_detections if d['is_confirmed']]
                if confirmed_detections:
                    best_marker = min(confirmed_detections, key=lambda x: x['distance'])
                
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
                    is_confirmed = detection['is_confirmed']
                    face_type = marker_annotation['face_type']
                    
                    # Draw coordinate axes for all markers
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
                    # Only show wireframe for the most confident (closest) confirmed marker
                    is_best_marker = (detection == best_marker)
                    if show_mesh and is_best_marker and is_confirmed:
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
                    y_offset = 30 + i * 120
                    
                    # Use different colors for best vs other markers
                    if is_best_marker and is_confirmed:
                        text_color = (0, 255, 0)  # Green for best confirmed marker
                        marker_status = " (BEST)"
                    elif is_confirmed:
                        text_color = (0, 255, 255)  # Cyan for other confirmed markers
                        marker_status = " (CONFIRMED)"
                    else:
                        text_color = (255, 255, 0)  # Yellow for holding markers
                        marker_status = " (HOLDING)"
                    
                    cv2.putText(frame, f"Marker ID: {marker_id} ({face_type}){marker_status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    cv2.putText(frame, f"Marker Pos: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                               (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    cv2.putText(frame, f"Distance: {distance:.3f}m", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Show object pose info only for the best marker
                    if is_best_marker and is_confirmed:
                        object_tvec, object_rvec = estimate_object_pose_from_marker((tvec, rvec), marker_annotation)
                        
                        # Convert rotation vector to RPY angles
                        rotation_matrix, _ = cv2.Rodrigues(object_rvec)
                        rpy = rotation_matrix_to_euler(rotation_matrix)
                        
                        cv2.putText(frame, f"Object Pos: ({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f})", 
                                   (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, f"Object RPY: ({np.degrees(rpy[0]):.1f}deg, {np.degrees(rpy[1]):.1f}deg, {np.degrees(rpy[2]):.1f}deg)", 
                                   (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Print to console for the best marker
                        print(f"\rBest Marker {marker_id} ({face_type}): Marker=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) | "
                              f"Object=({object_tvec[0]:.3f}, {object_tvec[1]:.3f}, {object_tvec[2]:.3f}) | "
                              f"RPY=({np.degrees(rpy[0]):.1f}°, {np.degrees(rpy[1]):.1f}°, {np.degrees(rpy[2]):.1f}°) | "
                              f"Mesh: {'ON' if show_mesh else 'OFF'}", end="", flush=True)
                
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
        cv2.imshow("Object Pose Estimation with Kalman Filtering", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_mesh = not show_mesh
            print(f"\nMesh display: {'ON' if show_mesh else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nObject pose estimation with Kalman filtering stopped.")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Object Pose Estimator with Kalman Filtering - Estimate 6D object pose from ArUco markers")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Model name to use (e.g., 'fork_orange_scaled70'). If not provided, interactive selection will be used.")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                       help="Path to data directory containing wireframe and aruco subdirectories")
    parser.add_argument("--list-models", "-l", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--camera-id", "-c", type=int, default=None,
                       help="Camera device ID to use (e.g., 0, 1, 8). If not provided, will scan and prompt.")
    
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
    
    # Run the object pose estimation with Kalman filtering
    detect_object_pose_with_kalman_filtering(model_name=args.model, data_dir=data_dir, camera_id=args.camera_id)

if __name__ == "__main__":
    main()
