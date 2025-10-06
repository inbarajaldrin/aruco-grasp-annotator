#!/usr/bin/env python3
"""
Annotate Grasp Points to All Markers
Takes grasp points from one marker and transforms them to be relative to ALL markers.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import argparse


def load_grasp_points(grasp_json_path):
    """Load grasp points from JSON file."""
    with open(grasp_json_path, 'r') as f:
        data = json.load(f)
    return data


def load_aruco_annotations(aruco_json_path):
    """Load ArUco marker annotations."""
    with open(aruco_json_path, 'r') as f:
        data = json.load(f)
    return data


def load_wireframe(wireframe_json_path):
    """Load wireframe data."""
    with open(wireframe_json_path, 'r') as f:
        data = json.load(f)
    return data


def get_marker_transform(marker_data):
    """
    Get the transformation matrix from marker frame to CAD center frame.
    
    Returns 4x4 transformation matrix.
    """
    # Get marker position relative to CAD center
    pos = marker_data['pose_relative_to_cad_center']['position']
    position = np.array([pos['x'], pos['y'], pos['z']])
    
    # Get marker rotation
    rot = marker_data['pose_relative_to_cad_center']['rotation']
    rotation = R.from_euler('xyz', [rot['roll'], rot['pitch'], rot['yaw']])
    
    # Build transformation matrix (marker to CAD center)
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = position
    
    return transform


def transform_point_between_markers(point, source_marker_data, target_marker_data):
    """
    Transform a point from source marker frame to target marker frame.
    
    Args:
        point: 3D point in source marker frame
        source_marker_data: Source marker annotation data
        target_marker_data: Target marker annotation data
    
    Returns:
        3D point in target marker frame
    """
    # Get transformations
    source_to_cad = get_marker_transform(source_marker_data)
    target_to_cad = get_marker_transform(target_marker_data)
    
    # Transform: source marker -> CAD center -> target marker
    # Point in source marker frame -> CAD center frame
    point_homogeneous = np.append(point, 1.0)
    point_in_cad = source_to_cad @ point_homogeneous
    
    # CAD center frame -> target marker frame
    cad_to_target = np.linalg.inv(target_to_cad)
    point_in_target = cad_to_target @ point_in_cad
    
    return point_in_target[:3]


def adjust_z_to_center(grasp_points, object_thickness):
    """
    Adjust Z-coordinate of grasp points to be at the center of the object.
    
    Args:
        grasp_points: List of grasp point dictionaries
        object_thickness: Thickness of the object (to divide by 2)
    
    Returns:
        List of adjusted grasp points
    """
    adjusted_points = []
    for gp in grasp_points:
        adjusted = gp.copy()
        adjusted['position'] = gp['position'].copy()
        # Set Z to half the thickness (center of object)
        adjusted['position']['z'] = object_thickness / 2.0
        adjusted_points.append(adjusted)
    
    return adjusted_points


def annotate_grasp_points_to_all_markers(object_name, source_marker_id, 
                                         object_thickness=None, data_dir="../../data"):
    """
    Main function to annotate grasp points relative to all markers.
    
    Args:
        object_name: Name of the object (e.g., "fork_orange_scaled70")
        source_marker_id: Marker ID used to generate the grasp points
        object_thickness: Thickness of the object (for Z adjustment). If None, auto-detect from ArUco annotations.
        data_dir: Path to data directory
    """
    data_path = Path(data_dir)
    
    # Load grasp points from source marker
    grasp_json = Path("outputs") / f"{object_name}_marker{source_marker_id}_grasp_points_3d.json"
    print(f"Loading grasp points from: {grasp_json}")
    grasp_data = load_grasp_points(grasp_json)
    
    # Load ArUco annotations
    aruco_json = data_path / "aruco" / f"{object_name}_aruco.json"
    print(f"Loading ArUco annotations from: {aruco_json}")
    aruco_data = load_aruco_annotations(aruco_json)
    
    # Load wireframe
    wireframe_json = data_path / "wireframe" / f"{object_name}_wireframe.json"
    print(f"Loading wireframe from: {wireframe_json}")
    wireframe_data = load_wireframe(wireframe_json)
    
    # Find source marker
    source_marker = None
    for marker in aruco_data['markers']:
        if marker['aruco_id'] == source_marker_id:
            source_marker = marker
            break
    
    if source_marker is None:
        raise ValueError(f"Source marker {source_marker_id} not found")
    
    # Auto-detect object thickness if not provided
    if object_thickness is None:
        if 'cad_object_info' in source_marker and 'dimensions' in source_marker['cad_object_info']:
            # Use height as thickness (assumes object is oriented with thickness in Z)
            object_thickness = source_marker['cad_object_info']['dimensions']['height']
            print(f"\n✓ Auto-detected object thickness from ArUco annotations: {object_thickness} m")
        else:
            raise ValueError("Object thickness not provided and could not be auto-detected from ArUco annotations")
    
    print(f"\nSource marker {source_marker_id} found")
    print(f"Grasp points from source: {len(grasp_data['grasp_points'])}")
    print(f"Object thickness: {object_thickness} m")
    
    # Adjust Z-coordinates to center of object
    print("\nAdjusting Z-coordinates to object center...")
    adjusted_grasp_points = adjust_z_to_center(
        grasp_data['grasp_points'], 
        object_thickness
    )
    
    # Convert to numpy array for easier manipulation
    source_points_3d = []
    for gp in adjusted_grasp_points:
        point = np.array([
            gp['position']['x'],
            gp['position']['y'],
            gp['position']['z']
        ])
        source_points_3d.append(point)
    
    # Transform grasp points to all markers
    print("\nTransforming grasp points to all markers...")
    all_markers_grasp_data = {
        "object_name": object_name,
        "display_name": object_name.replace('_', ' ').title(),
        "source_marker_id": source_marker_id,
        "object_thickness": object_thickness,
        "total_grasp_points": len(source_points_3d),
        "wireframe": {
            "vertices": wireframe_data['vertices'],
            "edges": wireframe_data['edges'],
            "mesh_info": wireframe_data['mesh_info']
        },
        "markers": []
    }
    
    for marker in aruco_data['markers']:
        marker_id = marker['aruco_id']
        print(f"  Processing marker {marker_id}...")
        
        # Transform each grasp point to this marker's frame
        grasp_points_in_marker = []
        for i, point in enumerate(source_points_3d):
            transformed_point = transform_point_between_markers(
                point, source_marker, marker
            )
            
            # Convert from centimeters to meters to match wireframe/ArUco scale
            grasp_points_in_marker.append({
                "id": i + 1,
                "position": {
                    "x": float(transformed_point[0] / 100.0),
                    "y": float(transformed_point[1] / 100.0),
                    "z": float(transformed_point[2] / 100.0)
                },
                "type": "center_point",
                "approach_vector": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0
                }
            })
        
        marker_grasp_data = {
            "aruco_id": marker_id,
            "size": marker.get('size', 0.02),  # Default size if not specified
            "pose_absolute": {
                "position": marker['pose_relative_to_cad_center']['position'],
                "rotation": marker['pose_relative_to_cad_center']['rotation']
            },
            "grasp_points": grasp_points_in_marker
        }
        
        all_markers_grasp_data["markers"].append(marker_grasp_data)
    
    # Save to grasp directory
    output_dir = data_path / "grasp"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{object_name}_grasp_points_all_markers.json"
    with open(output_file, 'w') as f:
        json.dump(all_markers_grasp_data, f, indent=2)
    
    print(f"\n✓ Saved grasp points for all markers: {output_file}")
    print(f"\nSummary:")
    print(f"  Object: {object_name}")
    print(f"  Source marker: {source_marker_id}")
    print(f"  Total grasp points: {len(source_points_3d)}")
    print(f"  Total markers: {len(all_markers_grasp_data['markers'])}")
    print(f"  Each marker now has {len(source_points_3d)} grasp points in its frame")
    
    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Annotate grasp points relative to all ArUco markers"
    )
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object name (e.g., fork_orange_scaled70)'
    )
    parser.add_argument(
        '--source-marker-id',
        type=int,
        required=True,
        help='Marker ID used to generate the grasp points'
    )
    parser.add_argument(
        '--object-thickness',
        type=float,
        required=False,
        default=None,
        help='Thickness of the object for Z-coordinate adjustment (auto-detects from ArUco annotations if not provided)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../../data',
        help='Path to data directory (default: ../../data)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ANNOTATE GRASP POINTS TO ALL MARKERS")
    print("="*60)
    
    output_file = annotate_grasp_points_to_all_markers(
        object_name=args.object,
        source_marker_id=args.source_marker_id,
        object_thickness=args.object_thickness,
        data_dir=args.data_dir
    )
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

