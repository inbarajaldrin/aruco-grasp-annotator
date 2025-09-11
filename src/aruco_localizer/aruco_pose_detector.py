import cv2
import cv2.aruco as aruco
import numpy as np

def detect_aruco_pose():
    """Detect 6D pose of ArUco marker ID 5 from 4x4_50 dictionary"""
    
    # Camera parameters (you can calibrate these for better accuracy)
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # ArUco parameters
    marker_size = 0.05  # 5cm marker size (adjust to your actual marker size)
    target_id = 5  # ID of the marker we want to detect
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Open camera
    cap = cv2.VideoCapture(8)  # Using camera ID 8 as specified
    if not cap.isOpened():
        print("Failed to open camera 8")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("ArUco Pose Detector Started")
    print("Looking for marker ID 5 from DICT_4X4_50")
    print("Press 'q' to quit")
    print("=" * 50)
    
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
            
            # Check if our target marker (ID 5) is detected
            if target_id in ids:
                # Find the index of our target marker
                target_idx = np.where(ids.flatten() == target_id)[0][0]
                target_corners = corners[target_idx]
                
                # Estimate pose
                object_points = np.array([
                    [-marker_size/2,  marker_size/2, 0],
                    [ marker_size/2,  marker_size/2, 0],
                    [ marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(object_points, target_corners[0], camera_matrix, dist_coeffs)
                
                if success:
                    # Draw coordinate axes
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Extract position (translation vector)
                    position = tvec.flatten()
                    
                    # Extract rotation (Euler angles from rotation matrix)
                    euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                    
                    # Calculate distance from camera
                    distance = np.linalg.norm(position)
                    
                    # Display pose information on frame
                    cv2.putText(frame, f"Marker ID: {target_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Position (x,y,z): ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Rotation (rx,ry,rz): ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f})", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {distance:.3f}m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Print to console for debugging
                    print(f"\rPosition: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} | "
                          f"Rotation: rx={euler_angles[0]:.1f}, ry={euler_angles[1]:.1f}, rz={euler_angles[2]:.1f} | "
                          f"Distance: {distance:.3f}m", end="", flush=True)
                else:
                    cv2.putText(frame, f"Marker ID {target_id} detected but pose estimation failed", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Looking for marker ID {target_id}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow("ArUco Pose Detection", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nArUco pose detection stopped.")

if __name__ == "__main__":
    detect_aruco_pose()
