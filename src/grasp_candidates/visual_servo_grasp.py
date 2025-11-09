#!/usr/bin/env python3
"""
Direct Object Movement - Native ROS2 Node
Read object poses from ObjectPoseArray and perform single direct movement to specific object by name
Includes calibration offset correction for accurate positioning
Supports grasp point selection from /grasp_points topic
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import from local action_libraries file
from action_libraries import hover_over_grasp, hover_arbitrary_orientation

# Import the new message types
try:
    from max_camera_msgs.msg import ObjectPoseArray
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs not found. Using geometry_msgs.PoseStamped as fallback.")
    ObjectPoseArray = None

# Import grasp points message type
try:
    from max_camera_msgs.msg import GraspPointArray, GraspPoint
except ImportError:
    # Fallback if the message type is not available
    print("Warning: max_camera_msgs GraspPointArray not found. Using geometry_msgs.PoseStamped as fallback.")
    GraspPointArray = None
    GraspPoint = None

class PoseKalmanFilter:
    """Kalman filter for pose estimation and smoothing"""
    
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
        self.state_dim = 12
        self.measurement_dim = 6  # [x, y, z, roll, pitch, yaw]
        
        # State vector
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10  # Initial covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * process_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Measurement matrix (we only measure position and orientation)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:6, :6] = np.eye(6)
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        dt = 1.0  # Time step (will be updated dynamically)
        self.F[:6, 6:] = np.eye(6) * dt
        
        self.initialized = False
        
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees"""
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            pitch = math.degrees(math.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        
        return [roll, pitch, yaw]
    
    def update(self, pose_msg, dt=1.0):
        """Update Kalman filter with new pose measurement"""
        # Extract measurement
        position = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        rpy = self.quaternion_to_rpy(
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        )
        
        measurement = np.array(position + rpy)
        
        # Update state transition matrix with current dt
        self.F[:6, 6:] = np.eye(6) * dt
        
        if not self.initialized:
            # Initialize state
            self.x[:6] = measurement
            self.initialized = True
            return self.x[:6], self.x[6:12]  # Return position and velocity
        
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        
        return self.x[:6], self.x[6:12]  # Return filtered position and velocity
    
    def get_filtered_pose(self):
        """Get current filtered pose"""
        if not self.initialized:
            return None, None
        return self.x[:6], self.x[6:12]

class DirectObjectMove(Node):
    def __init__(self, topic_name="/objects_poses_sim", object_name="blue_dot_0", height=None, movement_duration=10.0, target_xyz=None, target_xyzw=None, grasp_points_topic="/grasp_points", grasp_id=None, grasp_candidates_topic="/grasp_candidates", grasp_candidate_id=None, gripper_offset=0.115, orientation_offset=None):
        super().__init__('direct_object_move')
        
        self.topic_name = topic_name
        self.object_name = object_name
        self.height = height  # None means use offset, otherwise use exact height
        self.movement_duration = movement_duration  # Duration for IK movement
        self.target_xyz = target_xyz  # Optional target position [x, y, z]
        self.target_xyzw = target_xyzw  # Optional target orientation [x, y, z, w]
        self.grasp_points_topic = grasp_points_topic  # Topic for grasp points
        self.grasp_id = grasp_id  # Specific grasp point ID to use (legacy)
        self.grasp_candidates_topic = grasp_candidates_topic  # Topic for grasp candidates
        self.grasp_candidate_id = grasp_candidate_id  # Specific grasp candidate ID to use (new)
        self.last_target_pose = None
        self.position_threshold = 0.005  # 5mm
        self.angle_threshold = 2.0       # 2 degrees
        # Calibration offset to correct systematic detection bias
        self.calibration_offset_x = -0.0  # -0mm correction (move left)
        self.calibration_offset_y = -0.0  # +0mm correction (move forward)
        # Gripper offset distance from TCP to fingertips (along gripper Z-axis)
        self.object_to_ee_offset = gripper_offset  # Default: 0.115m = 11.5cm
        # Orientation offset quaternion [x, y, z, w] to apply to grasp candidate orientations
        # Default: [0, 1, 0, 0] = 180° rotation around Y-axis
        if orientation_offset is None:
            self.orientation_offset = np.array([0.0, 1.0, 0.0, 0.0])
        else:
            self.orientation_offset = np.array(orientation_offset)
        
        # Initialize Kalman filter
        self.kalman_filter = PoseKalmanFilter(process_noise=0.005, measurement_noise=0.05)
        self.last_update_time = None
        
        # Store latest grasp points and candidates
        self.latest_grasp_points = None
        self.selected_grasp_point = None
        self.latest_grasp_candidates = None
        self.selected_grasp_candidate = None
        
        # Store current end-effector pose
        self.current_ee_pose = None
        self.ee_pose_received = False
        
        # Subscribe to object poses topic
        # Use TFMessage (for /objects_poses_real topic which publishes TFMessage)
        self.pose_sub = self.create_subscription(
            TFMessage,
            topic_name,
            self.tf_message_callback,
            5  # Lower QoS to reduce update frequency
        )
        
        # Subscribe to end-effector pose topic
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            qos_profile
        )
        
        # Note: ObjectPoseArray subscription removed - topic publishes TFMessage format
        # If you need ObjectPoseArray support, use a different topic name
        
        # Subscribe to grasp candidates topic if grasp_candidate_id is provided
        if self.grasp_candidate_id is not None and GraspPointArray is not None:
            self.grasp_candidates_sub = self.create_subscription(
                GraspPointArray,
                grasp_candidates_topic,
                self.grasp_candidates_callback,
                5
            )
            self.get_logger().info(f"🎯 Grasp candidate mode: Looking for candidate_id {grasp_candidate_id} on topic {grasp_candidates_topic}")
            self.grasp_points_sub = None
        # Subscribe to grasp points topic if grasp_id is provided (legacy mode)
        elif self.grasp_id is not None and GraspPointArray is not None:
            self.grasp_points_sub = self.create_subscription(
                GraspPointArray,
                grasp_points_topic,
                self.grasp_points_callback,
                5
            )
            self.grasp_candidates_sub = None
            self.get_logger().info(f"🎯 Grasp point mode (legacy): Looking for grasp_id {grasp_id} on topic {grasp_points_topic}")
        else:
            self.grasp_points_sub = None
            self.grasp_candidates_sub = None
            if self.grasp_id is not None or self.grasp_candidate_id is not None:
                self.get_logger().warn(f"⚠️ Grasp mode requested but GraspPointArray not available. Falling back to object center.")
        
        # Add timer to control update frequency (every 2 seconds = 0.5Hz)
        self.update_timer = self.create_timer(3.0, self.timer_callback)
        self.latest_pose = None
        self.movement_completed = False  # Flag to track if movement has been completed
        self.should_exit = False  # Flag to control exit
        self.trajectory_in_progress = False  # Flag to track if trajectory is executing
        self.current_trajectory = None  # Store full trajectory for sequential execution
        self.current_trajectory_step = 0  # Track which step we're on (1 or 2)
        
        # Action client for trajectory execution
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.get_logger().info(f"🤖 Direct object movement started for object '{object_name}' on topic {topic_name}")
        if height is not None:
            self.get_logger().info(f"📏 Target height: {height}m (offset will be ignored)")
        else:
            self.get_logger().info(f"📏 Using {self.object_to_ee_offset*100:.1f}cm offset from object/grasp point")
        self.get_logger().info(f"⏱️ Movement duration: {movement_duration}s")
        if self.grasp_candidate_id is not None:
            self.get_logger().info(f"🎯 Grasp candidate mode: Using candidate_id {grasp_candidate_id} from topic {grasp_candidates_topic}")
        elif self.grasp_id is not None:
            self.get_logger().info(f"🎯 Grasp point mode (legacy): Using grasp_id {grasp_id} from topic {grasp_points_topic}")
        else:
            self.get_logger().info(f"🎯 Object center mode: Moving to object center")
        
    def quaternion_to_rpy(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw in degrees"""
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
        
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.degrees(math.copysign(math.pi / 2, sinp))
        else:
            pitch = math.degrees(math.asin(sinp))
        
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        
        return [roll, pitch, yaw]
    
    def poses_are_similar(self, position, rpy):
        """Check if pose is similar to last target"""
        if self.last_target_pose is None:
            return False
            
        last_pos, last_rpy = self.last_target_pose
        
        # Check position difference (only x, y)
        pos_diff = math.sqrt(
            (position[0] - last_pos[0])**2 +
            (position[1] - last_pos[1])**2
        )
        
        if pos_diff > self.position_threshold:
            return False
            
        # Check yaw difference
        angle_diff = abs(rpy[2] - last_rpy[2])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff <= self.angle_threshold
    
    def objects_poses_callback(self, msg):
        """Handle ObjectPoseArray message and find target object"""
        if ObjectPoseArray is None:
            return
            
        # Find the object with the specified name
        target_object = None
        for obj in msg.objects:
            if obj.object_name == self.object_name:
                target_object = obj
                break
        
        if target_object is not None:
            # Convert ObjectPose to PoseStamped for compatibility
            pose_stamped = PoseStamped()
            pose_stamped.header = target_object.header
            pose_stamped.pose = target_object.pose
            self.latest_pose = pose_stamped
        else:
            # Object not found in this message
            self.get_logger().warn(f"Object '{self.object_name}' not found in current message")
            self.latest_pose = None
    
    def tf_message_callback(self, msg):
        """Handle TFMessage and find target object by child_frame_id"""
        # Find the transform with matching child_frame_id (object name)
        target_transform = None
        for transform in msg.transforms:
            if transform.child_frame_id == self.object_name:
                target_transform = transform
                break
        
        if target_transform is not None:
            # Convert TransformStamped to PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header = target_transform.header
            pose_stamped.pose.position.x = target_transform.transform.translation.x
            pose_stamped.pose.position.y = target_transform.transform.translation.y
            pose_stamped.pose.position.z = target_transform.transform.translation.z
            pose_stamped.pose.orientation.x = target_transform.transform.rotation.x
            pose_stamped.pose.orientation.y = target_transform.transform.rotation.y
            pose_stamped.pose.orientation.z = target_transform.transform.rotation.z
            pose_stamped.pose.orientation.w = target_transform.transform.rotation.w
            self.latest_pose = pose_stamped
        else:
            # Object not found in this message
            self.latest_pose = None
    
    def pose_callback(self, msg):
        """Store latest pose message (fallback for PoseStamped)"""
        self.latest_pose = msg
    
    def grasp_points_callback(self, msg):
        """Handle GraspPointArray message and find target grasp point (legacy mode)"""
        if GraspPointArray is None:
            return
        
        # Store all grasp points
        self.latest_grasp_points = msg
        
        # Find the grasp point with the specified ID and object name
        target_grasp_point = None
        for grasp_point in msg.grasp_points:
            if (grasp_point.grasp_id == self.grasp_id and 
                grasp_point.object_name == self.object_name):
                target_grasp_point = grasp_point
                break
        
        if target_grasp_point is not None:
            self.selected_grasp_point = target_grasp_point
            self.get_logger().info(f"🎯 Found grasp point {self.grasp_id} for object '{self.object_name}'")
            # Unsubscribe after getting the grasp point once (simulation data is accurate)
            if self.grasp_points_sub is not None:
                self.destroy_subscription(self.grasp_points_sub)
                self.grasp_points_sub = None
        else:
            # Grasp point not found in this message
            self.get_logger().warn(f"Grasp point {self.grasp_id} for object '{self.object_name}' not found in current message")
            self.selected_grasp_point = None
    
    def grasp_candidates_callback(self, msg):
        """Handle GraspPointArray message and find target grasp candidate (new mode)"""
        if GraspPointArray is None:
            return
        
        # Store all grasp candidates
        self.latest_grasp_candidates = msg
        
        # Find the grasp candidate with the specified ID and object name
        target_grasp_candidate = None
        for grasp_candidate in msg.grasp_points:  # Note: candidates also use GraspPointArray format
            if (grasp_candidate.grasp_id == self.grasp_candidate_id and 
                grasp_candidate.object_name == self.object_name):
                target_grasp_candidate = grasp_candidate
                break
        
        if target_grasp_candidate is not None:
            self.selected_grasp_candidate = target_grasp_candidate
            self.get_logger().info(f"🎯 Found grasp candidate {self.grasp_candidate_id} for object '{self.object_name}'")
            self.get_logger().info(f"🎯 Candidate orientation (RPY): [{target_grasp_candidate.roll:.1f}, {target_grasp_candidate.pitch:.1f}, {target_grasp_candidate.yaw:.1f}]")
            # Unsubscribe after getting the grasp candidate once (simulation data is accurate)
            if self.grasp_candidates_sub is not None:
                self.destroy_subscription(self.grasp_candidates_sub)
                self.grasp_candidates_sub = None
        else:
            # Grasp candidate not found in this message
            self.get_logger().warn(f"Grasp candidate {self.grasp_candidate_id} for object '{self.object_name}' not found in current message")
            self.selected_grasp_candidate = None
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose data"""
        self.current_ee_pose = msg
        self.ee_pose_received = True
    
    def timer_callback(self):
        """Process pose and perform single direct movement to object"""
        if self.movement_completed:
            return
        
        # Don't send new trajectory if one is already in progress
        if self.trajectory_in_progress:
            self.get_logger().debug("Trajectory already in progress, skipping...")
            return
        
        # Wait for end-effector pose if not received yet
        if not self.ee_pose_received or self.current_ee_pose is None:
            self.get_logger().warn("Waiting for end-effector pose...")
            return
        
        # Get current end-effector position
        current_ee_position = np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])
        
        # Initialize variables for flexure joint behavior
        use_flexure_joint = False  # Default: not using flexure joint
        fingertips_position = None  # Will be set when using grasp candidates
        
        # Check if we have optional target position/orientation
        if self.target_xyz is not None and self.target_xyzw is not None:
            # Use provided target position and orientation
            object_position = np.array(self.target_xyz[:3])  # Take first 3 elements
            
            # Apply calibration offset to correct systematic detection bias (same as detected objects)
            object_position[0] += self.calibration_offset_x  # Correct X offset
            object_position[1] += self.calibration_offset_y  # Correct Y offset
            
            rpy = self.quaternion_to_rpy(
                self.target_xyzw[0], self.target_xyzw[1], 
                self.target_xyzw[2], self.target_xyzw[3]
            )
            self.get_logger().info(f"🎯 Using provided target position: {object_position} (with calibration offset applied) and orientation: {rpy}")
        elif self.selected_grasp_candidate is not None:
            # Use grasp candidate position (fingertips/grasp point) and orientation directly from the message
            # The grasp candidate position is the FIXED fingertips position (grasp point)
            fingertips_position = np.array([
                self.selected_grasp_candidate.pose.position.x,
                self.selected_grasp_candidate.pose.position.y,
                self.selected_grasp_candidate.pose.position.z
            ])
            
            # Apply calibration offset to correct systematic detection bias
            fingertips_position[0] += self.calibration_offset_x  # Correct X offset
            fingertips_position[1] += self.calibration_offset_y  # Correct Y offset
            
            # Use the provided RPY values directly from the grasp candidate message
            # This orientation will be used to calculate TCP position from fingertips
            roll = self.selected_grasp_candidate.roll
            pitch = self.selected_grasp_candidate.pitch
            yaw = self.selected_grasp_candidate.yaw
            
            # Convert RPY to quaternion
            r_base = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            quat_base = r_base.as_quat()  # Returns [x, y, z, w]
            
            # Apply orientation offset quaternion
            r_offset = R.from_quat(self.orientation_offset)
            r_final = r_offset * r_base  # Compose rotations: offset * base
            quat_final = r_final.as_quat()
            
            # Convert back to RPY
            rpy = r_final.as_euler('xyz', degrees=True).tolist()
            
            # For flexure joint behavior: fingertips are FIXED, TCP will be calculated from orientation
            object_position = fingertips_position  # Use fingertips as reference for logging
            
            self.get_logger().info(f"🎯 Using grasp candidate {self.grasp_candidate_id}")
            self.get_logger().info(f"   Fingertips position (FIXED): {fingertips_position} (with calibration offset applied)")
            self.get_logger().info(f"   Original orientation (RPY): [{roll:.1f}, {pitch:.1f}, {yaw:.1f}] degrees")
            self.get_logger().info(f"   Orientation offset (quat): [{self.orientation_offset[0]:.3f}, {self.orientation_offset[1]:.3f}, {self.orientation_offset[2]:.3f}, {self.orientation_offset[3]:.3f}]")
            self.get_logger().info(f"   Final orientation (RPY): [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] degrees")
            
            # Set flag to indicate we're using grasp candidate (for flexure joint behavior)
            use_flexure_joint = True
        elif self.selected_grasp_point is not None:
            # Use grasp point position and orientation directly from the message
            object_position = np.array([
                self.selected_grasp_point.pose.position.x,
                self.selected_grasp_point.pose.position.y,
                self.selected_grasp_point.pose.position.z
            ])
            
            # Apply calibration offset to correct systematic detection bias
            object_position[0] += self.calibration_offset_x  # Correct X offset
            object_position[1] += self.calibration_offset_y  # Correct Y offset
            
            # Use the provided RPY values directly from the grasp point message
            roll = self.selected_grasp_point.roll
            pitch = self.selected_grasp_point.pitch
            yaw = self.selected_grasp_point.yaw
            
            rpy = [roll, pitch, yaw]
            
            # Not using flexure joint for legacy grasp points
            use_flexure_joint = False
            
            self.get_logger().info(f"🎯 Using grasp point {self.grasp_id} position: {object_position} (with calibration offset applied)")
            self.get_logger().info(f"🎯 Grasp point orientation (RPY): [{roll:.1f}, {pitch:.1f}, {yaw:.1f}] degrees")
        elif self.latest_pose is not None:
            # Use detected object pose
            # Calculate time delta for Kalman filter
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 1.0  # Default time step
            self.last_update_time = current_time
            
            # Update Kalman filter with new measurement
            filtered_pose, velocity = self.kalman_filter.update(self.latest_pose, dt)
            
            if filtered_pose is None:
                return
                
            # Extract filtered position and orientation
            object_position = np.array(filtered_pose[:3])
            rpy = filtered_pose[3:6].tolist()
            
            # Apply calibration offset to correct systematic detection bias
            object_position[0] += self.calibration_offset_x  # Correct X offset
            object_position[1] += self.calibration_offset_y  # Correct Y offset
            
            # Not using flexure joint for detected object pose
            use_flexure_joint = False
            
            self.get_logger().info(f"🎯 Detected object at ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        else:
            # No target provided and no object detected
            self.get_logger().warn("No target position provided and no object detected")
            return
        
        # Calculate direction vector from object to current end-effector
        direction_vector = current_ee_position - object_position
        current_distance = np.linalg.norm(direction_vector)
        
        self.get_logger().info(f"📏 Current distance between object and EE: {current_distance*100:.2f} cm")
        self.get_logger().info(f"📍 Current EE position: ({current_ee_position[0]:.3f}, {current_ee_position[1]:.3f}, {current_ee_position[2]:.3f})")
        self.get_logger().info(f"📍 Object position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        
        # Determine approach direction
        # If using a grasp candidate or grasp point, use the approach vector from the orientation
        # Otherwise, use the direction from object to current EE (or default upward)
        approach_direction = None
        
        if self.selected_grasp_candidate is not None:
            # Use approach vector from grasp candidate orientation
            # The approach vector is the Z-axis of the grasp candidate frame
            # Convert quaternion to rotation matrix and extract Z-axis direction
            candidate_quat = np.array([
                self.selected_grasp_candidate.pose.orientation.x,
                self.selected_grasp_candidate.pose.orientation.y,
                self.selected_grasp_candidate.pose.orientation.z,
                self.selected_grasp_candidate.pose.orientation.w
            ])
            r = R.from_quat(candidate_quat)
            # Z-axis in grasp frame is [0, 0, 1], transform to base frame
            z_axis_grasp = np.array([0.0, 0.0, 1.0])
            approach_direction = r.apply(z_axis_grasp)
            approach_direction = approach_direction / np.linalg.norm(approach_direction)  # Normalize
            self.get_logger().info(f"🎯 Using grasp candidate approach direction: ({approach_direction[0]:.3f}, {approach_direction[1]:.3f}, {approach_direction[2]:.3f})")
        elif self.selected_grasp_point is not None:
            # Use approach vector from grasp point orientation (legacy mode)
            # The approach vector is the Z-axis of the grasp point frame
            # Convert quaternion to rotation matrix and extract Z-axis direction
            grasp_quat = np.array([
                self.selected_grasp_point.pose.orientation.x,
                self.selected_grasp_point.pose.orientation.y,
                self.selected_grasp_point.pose.orientation.z,
                self.selected_grasp_point.pose.orientation.w
            ])
            r = R.from_quat(grasp_quat)
            # Z-axis in grasp frame is [0, 0, 1], transform to base frame
            z_axis_grasp = np.array([0.0, 0.0, 1.0])
            approach_direction = r.apply(z_axis_grasp)
            approach_direction = approach_direction / np.linalg.norm(approach_direction)  # Normalize
            self.get_logger().info(f"🎯 Using grasp point approach direction: ({approach_direction[0]:.3f}, {approach_direction[1]:.3f}, {approach_direction[2]:.3f})")
        elif current_distance > 1e-6:
            # Use direction from object to current EE
            approach_direction = direction_vector / current_distance
            self.get_logger().info(f"🎯 Using current EE direction: ({approach_direction[0]:.3f}, {approach_direction[1]:.3f}, {approach_direction[2]:.3f})")
        else:
            # Default to upward direction
            approach_direction = np.array([0.0, 0.0, 1.0])
            self.get_logger().warn("Current distance is very small, using default upward direction")
        
        # If height is explicitly specified, use that exact height (ignore offset)
        if self.height is not None:
            # Use specified height directly, maintaining x-y position above/below object
            target_ee_position = np.array([
                object_position[0],
                object_position[1],
                self.height
            ])
            self.get_logger().info(f"📏 Using specified height={self.height:.3f}m (offset ignored)")
        else:
            # Flexure joint behavior: FINGERTIPS (grasp point) is FIXED, TCP moves in a sphere
            # The TCP position is calculated from the fingertips position using the orientation
            # This simulates a spherical flexure joint where the gripper can rotate around the fingertips
            # 
            # The gripper extends 0.115m from TCP along the gripper's Z-axis (points FROM TCP TO fingertips)
            # To keep fingertips at the grasp point, we calculate:
            #   TCP_position = fingertips - (gripper_z_axis_in_base_frame × 0.115m)
            #
            # Where gripper_z_axis is transformed to base frame using candidate's RPY orientation
            # This ensures the gripper fingertips are always at the grasp point while TCP moves in a sphere
            #
            # Mathematical model (Flexure Joint):
            #   Fingertips (grasp_point) = FIXED for all candidates
            #   TCP_position = fingertips - R(rpy) × [0, 0, 1] × 0.115m  [VARIES based on orientation]
            #   EE_orientation = from candidate RPY  [VARIES per candidate]
            #
            # Examples:
            #   RPY (0, 180, 0) -> gripper points down -> TCP = fingertips - [0, 0, -1]×0.115 = fingertips + [0, 0, 0.115]
            #   RPY (90, 0, 0) -> gripper horizontal -> TCP = fingertips - [1, 0, 0]×0.115 = fingertips + [-0.115, 0, 0]
            
            if use_flexure_joint:
                # Get fingertips position (grasp point) - this is FIXED
                # fingertips_position is already set from grasp candidate
                
                # Convert RPY to rotation matrix (from grasp candidate orientation)
                r_candidate = R.from_euler('xyz', rpy, degrees=True)
                
                # Gripper Z-axis vector in gripper frame (points FROM TCP TO fingertips)
                gripper_z_axis = np.array([0.0, 0.0, 1.0])
                
                # Transform gripper Z-axis to base frame using candidate orientation
                gripper_z_base = r_candidate.apply(gripper_z_axis)
                
                # TCP position = fingertips - (gripper_z_axis × offset)
                # This keeps fingertips FIXED while TCP moves in a sphere
                # Since gripper_z_base points FROM TCP TO fingertips, we subtract to get TCP position
                target_ee_position = fingertips_position - gripper_z_base * self.object_to_ee_offset
                
                self.get_logger().info(f"📏 Flexure joint: Fingertips FIXED at grasp point, TCP moves in sphere")
                self.get_logger().info(f"   Fingertips position: ({fingertips_position[0]:.3f}, {fingertips_position[1]:.3f}, {fingertips_position[2]:.3f}) m")
                self.get_logger().info(f"   Gripper Z-axis in base frame: ({gripper_z_base[0]:.3f}, {gripper_z_base[1]:.3f}, {gripper_z_base[2]:.3f})")
                self.get_logger().info(f"   TCP offset: {self.object_to_ee_offset*100:.1f}cm from fingertips")
            else:
                # Legacy behavior: use approach direction for non-grasp-candidate cases
                # This maintains backward compatibility
                if approach_direction is not None:
                    target_ee_position = object_position + approach_direction * self.object_to_ee_offset
                    self.get_logger().info(f"📏 Approach-based offset: TCP = object + approach_vector × {self.object_to_ee_offset*100:.1f}cm")
                else:
                    # Default: maintain current distance
                    target_ee_position = object_position + direction_vector / current_distance * self.object_to_ee_offset
                    self.get_logger().info(f"📏 Maintaining {self.object_to_ee_offset*100:.1f}cm offset from object/grasp point")
        
        # Verify the target distance (from fingertips to TCP)
        if use_flexure_joint:
            # Use fingertips_position for distance calculation when using grasp candidates
            calculated_distance = np.linalg.norm(target_ee_position - fingertips_position)
            self.get_logger().info(f"✅ Calculated TCP distance from fingertips: {calculated_distance*100:.2f} cm (should be {self.object_to_ee_offset*100:.1f}cm)")
        else:
            # For other cases, use object_position
            calculated_distance = np.linalg.norm(target_ee_position - object_position)
            self.get_logger().info(f"✅ Calculated target distance: {calculated_distance*100:.2f} cm")
        
        self.get_logger().info(f"🎯 Final target EE position: ({target_ee_position[0]:.3f}, {target_ee_position[1]:.3f}, {target_ee_position[2]:.3f})")
        
        # Create target pose with calculated position
        target_position = target_ee_position.tolist()
        target_pose = (target_position, rpy)
        
        # Use the calculated z-coordinate (which is either the specified height or the auto-calculated one)
        # Use hover_arbitrary_orientation for grasp candidates (supports arbitrary orientations)
        # Use hover_over_grasp for legacy grasp points (forces pitch=180)
        if self.selected_grasp_candidate is not None:
            self.get_logger().info(f"🤖 Using arbitrary orientation mode for grasp candidate")
            trajectory = hover_arbitrary_orientation(target_pose, target_ee_position[2], self.movement_duration)
        else:
            self.get_logger().info(f"🤖 Using legacy top-down mode (pitch=180)")
            trajectory = hover_over_grasp(target_pose, target_ee_position[2], self.movement_duration)
        
        # Store trajectory for sequential execution
        self.current_trajectory = trajectory
        self.current_trajectory_step = 1  # Start with step 1
        
        # Execute trajectory
        self.trajectory_in_progress = True  # Mark trajectory as in progress
        self.execute_trajectory(trajectory, step=1)
        # Don't set movement_completed or should_exit here - wait for trajectory completion
    
    def execute_trajectory(self, trajectory, step=1):
        """Execute trajectory using ROS2 action
        
        Args:
            trajectory: Dictionary with 'traj1' and optionally 'traj2' keys
            step: Which trajectory step to execute (1 or 2)
        """
        try:
            traj_key = f'traj{step}'
            if traj_key not in trajectory or not trajectory[traj_key]:
                if step == 1:
                    self.get_logger().error("No trajectory found")
                    return
                else:
                    # No traj2, so we're done after traj1
                    self.get_logger().info("Single-step trajectory (no reorientation step)")
                    return
            
            point = trajectory[traj_key][0]
            positions = point['positions']
            duration = point['time_from_start'].sec
            
            if step == 1:
                self.get_logger().info(f"🤖 Step {step}: Moving to position with gripper pointing down...")
            else:
                self.get_logger().info(f"🤖 Step {step}: Reorienting to target orientation (spherical joint rotation)...")
            
            # Create trajectory message
            traj_msg = JointTrajectory()
            traj_msg.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
            ]
            
            traj_point = JointTrajectoryPoint()
            traj_point.positions = positions
            traj_point.velocities = [0.0] * 6
            traj_point.time_from_start = Duration(sec=duration)
            traj_msg.points.append(traj_point)
            
            # Create and send goal
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)
            
            self.get_logger().info("Sending trajectory...")
            self._send_goal_future = self.action_client.send_goal_async(goal)
            self._send_goal_future.add_done_callback(self.goal_response)
            
        except Exception as e:
            self.get_logger().error(f"❌ Trajectory execution error: {e}")
            self.trajectory_in_progress = False  # Clear flag on error
            self.movement_completed = True
            self.should_exit = True

    def goal_response(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            # Set exit flags if goal is rejected
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True
            return

        self.get_logger().info("Trajectory goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        """Handle goal result"""
        result = future.result()
        
        if result.status == 4:  # SUCCEEDED
            if self.current_trajectory_step == 1:
                self.get_logger().info("✅ Step 1 completed: Position reached with gripper pointing down")
                
                # Check if there's a second trajectory step (reorientation)
                if self.current_trajectory and 'traj2' in self.current_trajectory and self.current_trajectory['traj2']:
                    # Execute step 2: Reorient to target orientation
                    self.current_trajectory_step = 2
                    self.get_logger().info("🔄 Starting Step 2: Reorienting to target orientation...")
                    self.execute_trajectory(self.current_trajectory, step=2)
                    return  # Don't exit yet, wait for step 2
                else:
                    # No step 2, we're done
                    self.trajectory_in_progress = False
                    self.movement_completed = True
                    self.should_exit = True
                    self.get_logger().info("✅ Direct movement completed. Exiting.")
            else:
                # Step 2 completed
                self.get_logger().info("✅ Step 2 completed: Reoriented to target orientation")
                self.trajectory_in_progress = False
                self.movement_completed = True
                self.should_exit = True
                self.get_logger().info("✅ Direct movement completed. Exiting.")
        else:
            self.get_logger().error(f"Trajectory failed with status: {result.status}")
            # Set exit flags on failure
            self.trajectory_in_progress = False
            self.movement_completed = True
            self.should_exit = True
            self.get_logger().error("❌ Movement failed. Exiting.")


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct Object Movement Node')
    parser.add_argument('--topic', type=str, default="/objects_poses_real", 
                       help='Topic name for object poses subscription')
    parser.add_argument('--object-name', type=str, default="fork_orange_scaled70",
                       help='Name of the object to move to (e.g., blue_dot_0, red_dot_0)')
    parser.add_argument('--height', type=float, default=None,
                       help='Hover height in meters (if not specified, will use 5.5cm offset from object/grasp point)')
    parser.add_argument('--movement-duration', type=float, default=10.0,
                       help='Duration for the movement in seconds (default: 10.0)')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                       help='Optional target position [x, y, z] in meters')
    parser.add_argument('--target-xyzw', type=float, nargs=4, default=None,
                       help='Optional target orientation [x, y, z, w] quaternion')
    parser.add_argument('--grasp-points-topic', type=str, default="/grasp_points",
                       help='Topic name for grasp points subscription (legacy mode)')
    parser.add_argument('--grasp-id', type=int, default=None,
                       help='Specific grasp point ID to use (legacy mode - forces pitch=180, top-down only)')
    parser.add_argument('--grasp-candidates-topic', type=str, default="/grasp_candidates",
                       help='Topic name for grasp candidates subscription (new mode with arbitrary orientations)')
    parser.add_argument('--grasp-candidate-id', type=int, default=None,
                       help='Specific grasp candidate ID to use (new mode - supports arbitrary orientations)')
    parser.add_argument('--gripper-offset', type=float, default=0.115,
                       help='Gripper offset distance from TCP to fingertips in meters (default: 0.115)')
    parser.add_argument('--orientation-offset', type=float, nargs=4, default=[0.0, 1.0, 0.0, 0.0],
                       help='Orientation offset quaternion [x, y, z, w] to apply to grasp candidate orientations (default: [0, 1, 0, 0] = 180° rotation around Y-axis)')
    
    # Parse arguments from sys.argv if args is None
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    rclpy.init(args=None)
    node = DirectObjectMove(topic_name=args.topic, object_name=args.object_name, 
                      height=args.height, movement_duration=args.movement_duration,
                      target_xyz=args.target_xyz, target_xyzw=args.target_xyzw,
                      grasp_points_topic=args.grasp_points_topic, grasp_id=args.grasp_id,
                      grasp_candidates_topic=args.grasp_candidates_topic, 
                      grasp_candidate_id=args.grasp_candidate_id,
                      gripper_offset=args.gripper_offset,
                      orientation_offset=args.orientation_offset)
    
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Direct movement stopped by user")
    except Exception as e:
        node.get_logger().error(f"Direct movement error: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            # Ignore shutdown errors
            pass

if __name__ == '__main__':
    main()
