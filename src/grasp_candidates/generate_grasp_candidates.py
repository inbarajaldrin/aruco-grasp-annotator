#!/usr/bin/env python3
"""
Generate standardized 6-DoF grasp candidates from filtered grasp points.

Spec: src/grasp_candidates/GRASP_CANDIDATE_SCHEMA.md  (schema_version 2)

For each grasp point we emit ONE candidate per non-null gripper-closing axis found in
``grasp_validity`` (validity-driven, top-down). The candidate carries the full gripper-TCP
orientation in the object/CAD-center frame:

    q_cand(V, R) = avq(V) * Rz(R) * q_conv

  * V  = approach vector (top-down V=[0,0,1] for flat objects)
  * R  = in-plane rotation selecting the closing axis: 0deg -> object X, 90deg -> object Y
  * q_conv = fixed gripper mount flip, rpy[180,0,180] = quat (0,1,0,0)

Composing with the live object pose (done by the publisher) gives the gripper command directly:
    q_world = q_obj_world (x) q_cand
which is numerically identical to move_to_grasp's verified face_down(object_yaw + R) target.

Input : data/grasp_points/<object>_grasp_points.json   (has position + approach_vector + grasp_validity)
Output: data/grasp_candidates/<object>_grasp_candidates.json
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.spatial.transform import Rotation

# --- standardized constants (see GRASP_CANDIDATE_SCHEMA.md) -----------------------------------
Q_CONV = Rotation.from_euler('xyz', [180.0, 0.0, 180.0], degrees=True)   # gripper mount flip, quat (0,1,0,0)
DEFAULT_STANDOFF_M = 0.115                                                # runtime pre-grasp standoff along V

# direction_id encodes the closing axis (preserves publisher addressing grasp_id = gp_id*100 + direction_id)
AXIS_TO_DIRECTION = {"x": 1, "y": 2}
AXIS_TO_INPLANE_DEG = {"x": 0.0, "y": 90.0}
AXIS_TO_VALIDITY_KEY = {"x": "x_axis_gripper_width_mm", "y": "y_axis_gripper_width_mm"}


def approach_vector_to_quaternion(approach_vec: Dict[str, float]) -> Rotation:
    """avq(V): rotation aligning the local +Z axis with the approach vector V.

    For top-down V=[0,0,1] this is identity. (Same construction the pipeline used before.)
    """
    approach = np.array([approach_vec['x'], approach_vec['y'], approach_vec['z']], dtype=float)
    approach = approach / np.linalg.norm(approach)
    z_ref = np.array([0.0, 0.0, 1.0])
    if abs(approach[2]) < 0.9:
        x_axis = np.cross(z_ref, approach)
        x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 1e-6 else np.array([1.0, 0.0, 0.0])
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(approach, x_axis); y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, approach); x_axis /= np.linalg.norm(x_axis)
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, approach]))


def candidate_quaternion(approach_vec: Dict[str, float], in_plane_deg: float) -> Rotation:
    """Full gripper-TCP orientation in object frame: avq(V) * Rz(R) * q_conv."""
    return approach_vector_to_quaternion(approach_vec) * Rotation.from_euler('z', in_plane_deg, degrees=True) * Q_CONV


def quat_dict(r: Rotation) -> Dict[str, float]:
    q = r.as_quat()  # [x, y, z, w]
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def rpy_dict(r: Rotation) -> Dict[str, float]:
    rpy = r.as_euler('xyz', degrees=True)
    rpy = [((a + 180.0) % 360.0) - 180.0 for a in rpy]
    return {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])}


def load_gripper(input_dir: Path) -> Dict[str, Any]:
    f = input_dir / "gripper.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"type": "parallel", "max_width_mm": 100, "clearance_mm": 14, "tip_thickness_mm": 20}


def build_candidates_for_point(grasp_point: Dict[str, Any], gripper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Emit one candidate per non-null validity axis (top-down, validity-driven)."""
    gp_id = grasp_point.get('id', 0)
    pos = grasp_point.get('position', {})
    approach_vec = grasp_point.get('approach_vector', {"x": 0.0, "y": 0.0, "z": 1.0})
    validity = grasp_point.get('grasp_validity', {}) or {}
    max_w = float(gripper.get("max_width_mm", 100))

    out: List[Dict[str, Any]] = []
    for axis in ("x", "y"):
        width = validity.get(AXIS_TO_VALIDITY_KEY[axis])
        if width is None:
            continue  # axis filtered out (no valid grasp / exceeds gripper) -> no candidate
        in_plane = AXIS_TO_INPLANE_DEG[axis]
        r_cand = candidate_quaternion(approach_vec, in_plane)
        out.append({
            "grasp_point_id": gp_id,
            "direction_id": AXIS_TO_DIRECTION[axis],     # 1=X-close, 2=Y-close (opaque variant id)
            "approach_name": "top",
            "approach_vector": {"x": float(approach_vec['x']), "y": float(approach_vec['y']), "z": float(approach_vec['z'])},
            "in_plane_rotation_deg": in_plane,
            "closing_axis": axis,
            "width_mm": round(min(float(width), max_w), 1),
            "standoff_m": DEFAULT_STANDOFF_M,
            "grasp_candidate_position": {"x": float(pos.get('x', 0.0)), "y": float(pos.get('y', 0.0)), "z": float(pos.get('z', 0.0))},
            "approach_quaternion": quat_dict(r_cand),    # FULL TCP-in-object (face-down), execution-ready
            "approach_rpy": rpy_dict(r_cand),
        })
    return out


def generate_for_file(grasp_file: Path, output_dir: Path, gripper: Dict[str, Any]) -> int:
    data = json.loads(grasp_file.read_text())
    object_name = data.get('object_name', grasp_file.stem.replace('_grasp_points', ''))
    grasp_points = data.get('grasp_points', [])

    candidates: List[Dict[str, Any]] = []
    for gp in grasp_points:
        candidates.extend(build_candidates_for_point(gp, gripper))

    out = {
        "object_name": object_name,
        "schema_version": 2,
        "coordinate_frame": data.get('coordinate_frame', 'cad_center'),
        "gripper": {
            "max_width_mm": gripper.get("max_width_mm"),
            "clearance_mm": gripper.get("clearance_mm"),
            "tip_thickness_mm": gripper.get("tip_thickness_mm"),
        },
        "total_grasp_candidates": len(candidates),
        "grasp_candidates": candidates,
    }
    out_path = output_dir / f"{object_name}_grasp_candidates.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✅ {object_name}: {len(candidates)} candidates ({len(grasp_points)} points) -> {out_path.name}")
    return len(candidates)


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    grasp_points_dir = project_root / "data" / "grasp_points"
    output_dir = project_root / "data" / "grasp_candidates"
    input_dir = project_root / "assets" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    gripper = load_gripper(input_dir)
    print("🚀 Generating standardized grasp candidates (schema_version 2)")
    print(f"📁 Input : {grasp_points_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Gripper: {gripper}")

    grasp_files = sorted(grasp_points_dir.glob("*_grasp_points.json"))
    if not grasp_files:
        print(f"\n❌ No grasp points files found in {grasp_points_dir}")
        return

    print(f"\n📋 Found {len(grasp_files)} grasp points file(s)")
    total = 0
    for f in grasp_files:
        try:
            total += generate_for_file(f, output_dir, gripper)
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n✅ Complete! {total} candidates across {len(grasp_files)} objects -> {output_dir}")


if __name__ == "__main__":
    main()
