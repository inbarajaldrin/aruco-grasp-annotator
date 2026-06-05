"""
Headless grasp-point pipeline: chains the three UI steps into one call.

  step1  CADToGraspPipeline.run(object, marker_id)  -> render top-down + detect 2D grasp
         points + map to 3D marker-relative  (outputs/{obj}_marker{id}_grasp_points_3d.json)
  filter GraspFilter (optional, on by default)      -> gripper-constraint filtering + validity
         metadata, in place on the step1 json (reuses step1's detected regions; no re-render)
  step2  annotate_grasp_points_to_all_markers(...)  -> transform to CAD-centre frame + carry
         every marker's pose  (data_dir/grasp_points/{obj}_grasp_points.json)

Depends on the object's aruco (marker frame + auto thickness) and wireframe (px->mm), so those
must exist under data_dir. Boards are not grasped — run this for graspable objects only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical filter params (the FilterRequest defaults the UI uses).
DEFAULT_FILTER: dict[str, Any] = {
    "gripper_max_width_mm": 100.0,
    "grasp_clearance_mm": 14.0,
    "gripper_tip_thickness_mm": 20.0,
    "max_gap_px": 20,
    "symmetry_tolerance_mm": 10.0,
    "check_x_axis": True,
    "check_y_axis": True,
}


def top_marker_id(data_dir: str | Path, object_name: str) -> int:
    """The marker on the top (+z) face — the natural viewpoint for top-down grasp detection."""
    aruco = json.loads((Path(data_dir) / "aruco" / f"{object_name}_aruco.json").read_text())
    markers = aruco.get("markers", [])
    for m in markers:
        if m.get("face_type") == "top":
            return int(m["aruco_id"])
    return int(markers[0]["aruco_id"])  # fallback: first marker


def _filter_in_place(step1_json: Path, regions, data_dir: Path, object_name: str, filter_params):
    from ..utils.grasp_filter import GraspFilter

    fp = {**DEFAULT_FILTER, **(filter_params or {})}
    grasp_data = json.loads(step1_json.read_text())
    wireframe = json.loads((data_dir / "wireframe" / f"{object_name}_wireframe.json").read_text())
    filter_input = {
        "grasp_points": grasp_data.get("grasp_points", []),
        "wireframe": {"vertices": wireframe["vertices"]},
    }
    grasp_filter = GraspFilter(
        gripper_max_width_mm=fp["gripper_max_width_mm"],
        grasp_clearance_mm=fp["grasp_clearance_mm"],
        gripper_tip_thickness_mm=fp["gripper_tip_thickness_mm"],
        max_gap_px=fp["max_gap_px"],
        symmetry_tolerance_mm=fp["symmetry_tolerance_mm"],
    )
    result = grasp_filter.filter_grasp_points(
        filter_input, regions, check_x_axis=fp["check_x_axis"], check_y_axis=fp["check_y_axis"]
    )
    validity = {r["grasp_id"]: r for r in result["results"]}
    points = []
    for gp in result["filtered_grasp_points"]:
        info = validity.get(gp["id"], {})
        enhanced = gp.copy()
        enhanced["grasp_validity"] = {
            "x_axis_gripper_width_mm": info.get("x_axis_gripper_width_mm"),
            "y_axis_gripper_width_mm": info.get("y_axis_gripper_width_mm"),
        }
        points.append(enhanced)
    grasp_data["grasp_points"] = points
    grasp_data["total_points"] = len(points)
    grasp_data["filter_applied"] = True
    grasp_data["filter_params"] = result["filter_params"]
    step1_json.write_text(json.dumps(grasp_data, indent=2))


def generate_grasp_points(
    object_name: str,
    *,
    data_dir: str | Path,
    outputs_dir: str | Path,
    marker_id: int | None = None,
    apply_filter: bool = True,
    filter_params: dict | None = None,
    object_thickness: float | None = None,
    camera_distance: float = 0.5,
    min_area_threshold: int = 1000,
    use_mtl_color: bool = False,
) -> str:
    """Run step1 -> [filter] -> step2 headlessly. Returns the grasp_points json path."""
    from .annotation_transformer import annotate_grasp_points_to_all_markers
    from .pipeline import CADToGraspPipeline

    data_dir = Path(data_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if marker_id is None:
        marker_id = top_marker_id(data_dir, object_name)

    pipeline = CADToGraspPipeline(data_dir=str(data_dir), outputs_dir=str(outputs_dir))
    result = pipeline.run(
        object_name,
        marker_id,
        camera_distance=camera_distance,
        min_area_threshold=min_area_threshold,
        use_mtl_color=use_mtl_color,
    )
    step1_json = Path(result["output_json"])

    if apply_filter:
        _filter_in_place(step1_json, result["regions"], data_dir, object_name, filter_params)

    return annotate_grasp_points_to_all_markers(
        object_name,
        marker_id,
        str(step1_json),
        object_thickness=object_thickness,
        data_dir=str(data_dir),
    )
