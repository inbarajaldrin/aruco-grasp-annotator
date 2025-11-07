#!/usr/bin/env python3
"""
Generate grasp candidates with all possible approach vector orientations.

This script reads grasp points from JSON files in the data/grasp/ folder
and generates all possible approach vector orientations for each grasp point,
saving them as separate JSON files in the data/grasp_candidates/ folder.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Define all possible approach vectors (normalized directions)
# 6 cardinal directions + 12 intermediate directions = 18 total

# Normalization factor for intermediate directions (45 degrees between axes)
sqrt2_inv = 1.0 / math.sqrt(2.0)

APPROACH_VECTORS = [
    # 6 Cardinal directions
    {"x": 0.0, "y": 0.0, "z": 1.0, "name": "top"},           # 1: From above (+Z)
    {"x": 0.0, "y": 0.0, "z": -1.0, "name": "bottom"},      # 2: From below (-Z)
    {"x": 1.0, "y": 0.0, "z": 0.0, "name": "right"},        # 3: From right (+X)
    {"x": -1.0, "y": 0.0, "z": 0.0, "name": "left"},        # 4: From left (-X)
    {"x": 0.0, "y": 1.0, "z": 0.0, "name": "front"},        # 5: From front (+Y)
    {"x": 0.0, "y": -1.0, "z": 0.0, "name": "back"},        # 6: From back (-Y)
    
    # 12 Intermediate directions (45 degrees between axes)
    # Between +X and +Y
    {"x": sqrt2_inv, "y": sqrt2_inv, "z": 0.0, "name": "right_front"},      # 7
    # Between +X and -Y
    {"x": sqrt2_inv, "y": -sqrt2_inv, "z": 0.0, "name": "right_back"},       # 8
    # Between -X and +Y
    {"x": -sqrt2_inv, "y": sqrt2_inv, "z": 0.0, "name": "left_front"},      # 9
    # Between -X and -Y
    {"x": -sqrt2_inv, "y": -sqrt2_inv, "z": 0.0, "name": "left_back"},       # 10
    
    # Between +X and +Z
    {"x": sqrt2_inv, "y": 0.0, "z": sqrt2_inv, "name": "right_top"},         # 11
    # Between +X and -Z
    {"x": sqrt2_inv, "y": 0.0, "z": -sqrt2_inv, "name": "right_bottom"},      # 12
    # Between -X and +Z
    {"x": -sqrt2_inv, "y": 0.0, "z": sqrt2_inv, "name": "left_top"},         # 13
    # Between -X and -Z
    {"x": -sqrt2_inv, "y": 0.0, "z": -sqrt2_inv, "name": "left_bottom"},     # 14
    
    # Between +Y and +Z
    {"x": 0.0, "y": sqrt2_inv, "z": sqrt2_inv, "name": "front_top"},         # 15
    # Between +Y and -Z
    {"x": 0.0, "y": sqrt2_inv, "z": -sqrt2_inv, "name": "front_bottom"},     # 16
    # Between -Y and +Z
    {"x": 0.0, "y": -sqrt2_inv, "z": sqrt2_inv, "name": "back_top"},         # 17
    # Between -Y and -Z
    {"x": 0.0, "y": -sqrt2_inv, "z": -sqrt2_inv, "name": "back_bottom"},      # 18
]

def calculate_grasp_candidate_position(grasp_point_pos: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the grasp candidate position from the grasp point position.
    The candidate position is the same for all approach directions of the same grasp point.
    
    For now, the candidate position is the same as the grasp point position.
    The offset of 0.115m is applied at runtime when moving the end-effector.
    
    Args:
        grasp_point_pos: Dictionary with 'x', 'y', 'z' keys for grasp point position
    
    Returns:
        Dictionary with 'x', 'y', 'z' keys for grasp candidate position
    """
    # The candidate position is the same as the grasp point position
    # The 0.115m offset is applied at runtime along the approach direction
    return {
        "x": float(grasp_point_pos['x']),
        "y": float(grasp_point_pos['y']),
        "z": float(grasp_point_pos['z'])
    }


def generate_grasp_candidates_for_file(grasp_file: Path, output_dir: Path):
    """
    Generate grasp candidates for all grasp points in a single JSON file.
    Creates one output file per object containing all grasp candidates.
    
    Args:
        grasp_file: Path to the input grasp points JSON file
        output_dir: Directory to save the output JSON files
    
    Returns:
        Number of grasp candidates generated
    """
    print(f"\n📂 Processing: {grasp_file.name}")
    
    # Read the grasp points file
    with open(grasp_file, 'r') as f:
        grasp_data = json.load(f)
    
    object_name = grasp_data.get('object_name', 'unknown')
    grasp_points = grasp_data.get('grasp_points', [])
    
    print(f"  Found {len(grasp_points)} grasp points for object: {object_name}")
    
    # Create output structure with all grasp candidates
    output_data = {
        "object_name": object_name,
        "total_grasp_candidates": 0,
        "grasp_candidates": []
    }
    
    # Generate candidates for each grasp point
    for grasp_point in grasp_points:
        grasp_point_id = grasp_point.get('id', 0)
        grasp_point_pos = grasp_point.get('position', {})
        
        # Calculate grasp candidate position once for this grasp point
        # (same for all directions)
        candidate_pos = calculate_grasp_candidate_position(grasp_point_pos)
        
        # Generate a candidate for each approach vector
        for direction_id, approach_vec in enumerate(APPROACH_VECTORS, start=1):
            # Create the candidate JSON structure
            candidate_data = {
                "grasp_point_id": grasp_point_id,
                "direction_id": direction_id,
                "grasp_candidate_position": candidate_pos,
                "approach_vector": {
                    "x": approach_vec["x"],
                    "y": approach_vec["y"],
                    "z": approach_vec["z"]
                }
            }
            
            output_data["grasp_candidates"].append(candidate_data)
            output_data["total_grasp_candidates"] += 1
    
    # Generate output filename: {object_name}_grasp_candidates.json
    output_filename = f"{object_name}_grasp_candidates.json"
    output_path = output_dir / output_filename
    
    # Save the candidate JSON file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    total_candidates = output_data["total_grasp_candidates"]
    print(f"  ✅ Generated {total_candidates} grasp candidates ({len(grasp_points)} points × {len(APPROACH_VECTORS)} directions)")
    print(f"  💾 Saved to: {output_filename}")
    return total_candidates


def main():
    """Main entry point."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    grasp_dir = project_root / "data" / "grasp"
    output_dir = project_root / "data" / "grasp_candidates"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generating grasp candidates with all possible approach vectors")
    print(f"📁 Input directory: {grasp_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📝 Note: Candidate position is the same for all directions of the same grasp point")
    print(f"🧭 Approach directions: {len(APPROACH_VECTORS)}")
    print("   " + ", ".join([f"{i+1}. {v['name']}" for i, v in enumerate(APPROACH_VECTORS)]))
    
    # Find all grasp points JSON files
    grasp_files = list(grasp_dir.glob("*_grasp_points_all_markers.json"))
    
    if not grasp_files:
        print(f"\n❌ No grasp points files found in {grasp_dir}")
        print("   Expected files matching pattern: *_grasp_points_all_markers.json")
        return
    
    print(f"\n📋 Found {len(grasp_files)} grasp points file(s)")
    
    # Process each file
    total_candidates = 0
    for grasp_file in sorted(grasp_files):
        try:
            candidates = generate_grasp_candidates_for_file(grasp_file, output_dir)
            total_candidates += candidates
        except Exception as e:
            print(f"  ❌ Error processing {grasp_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Complete! Generated {total_candidates} total grasp candidates")
    print(f"📁 Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()

