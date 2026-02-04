"""Grasp point filtering routes for Grasp Points Annotator."""

import base64
import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Body, HTTPException

from shared.fastapi_utils import get_data_dir

from ..core.pipeline import CADToGraspPipeline
from ..utils.grasp_filter import GraspFilter
from ..utils.region_detector import visualize_center_points_only

router = APIRouter(prefix="/api")

# Directory paths
DATA_DIR = get_data_dir(Path(__file__).parent.parent / "app.py")
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


@router.post("/filter")
async def filter_grasp_points(data: dict[str, Any] = Body(...)):
    """
    Filter grasp points based on gripper constraints.
    Updates the grasp points file with only the filtered grasp points.
    """
    try:
        object_name = data.get("object_name")
        marker_id = data.get("marker_id")

        # Filter parameters (with defaults from grasp_filter.py)
        gripper_max_width_mm = data.get("gripper_max_width_mm", 70.0)
        gripper_half_open_width_mm = data.get("gripper_half_open_width_mm", 30.0)
        gripper_tip_thickness_mm = data.get("gripper_tip_thickness_mm", 20.0)
        max_gap_px = data.get("max_gap_px", 20)
        symmetry_tolerance_mm = data.get("symmetry_tolerance_mm", 10.0)
        check_x_axis = data.get("check_x_axis", True)
        check_y_axis = data.get("check_y_axis", True)

        if not object_name or marker_id is None:
            raise HTTPException(status_code=400, detail="object_name and marker_id are required")

        # Load grasp points JSON from step 1
        grasp_points_json = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_grasp_points_3d.json"
        if not grasp_points_json.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Grasp points file not found. Please run Step 1 first: {grasp_points_json}"
            )

        with open(grasp_points_json, 'r') as f:
            grasp_data = json.load(f)

        # Run region detection to get regions for filtering
        pipeline = CADToGraspPipeline(
            data_dir=str(DATA_DIR.resolve()),
            outputs_dir=str(OUTPUTS_DIR.resolve())
        )

        rendered_image_path = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_topdown.png"
        if not rendered_image_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Rendered image not found. Please run Step 1 first."
            )

        _, regions, _ = pipeline.detect_grasp_points_2d(
            rendered_image_path, object_name, marker_id, min_area_threshold=1000
        )

        # Load wireframe data for pixel-to-mm conversion
        wireframe_json = DATA_DIR / "wireframe" / f"{object_name}_wireframe.json"
        if not wireframe_json.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Wireframe file not found: {wireframe_json}"
            )

        with open(wireframe_json, 'r') as f:
            wireframe_data = json.load(f)

        # Create filter input structure
        filter_grasp_data = {
            'grasp_points': grasp_data.get('grasp_points', []),
            'wireframe': {
                'vertices': wireframe_data['vertices']
            }
        }

        # Initialize and run filter
        grasp_filter = GraspFilter(
            gripper_max_width_mm=gripper_max_width_mm,
            gripper_half_open_width_mm=gripper_half_open_width_mm,
            gripper_tip_thickness_mm=gripper_tip_thickness_mm,
            max_gap_px=max_gap_px,
            symmetry_tolerance_mm=symmetry_tolerance_mm
        )

        filter_result = grasp_filter.filter_grasp_points(
            filter_grasp_data,
            regions,
            check_x_axis=check_x_axis,
            check_y_axis=check_y_axis
        )

        # Update grasp data with filtered points and validity metadata
        validity_map = {r['grasp_id']: r for r in filter_result['results']}
        filtered_points_with_metadata = []

        for gp in filter_result['filtered_grasp_points']:
            grasp_id = gp['id']
            validity_info = validity_map.get(grasp_id, {})
            enhanced_gp = gp.copy()
            enhanced_gp['grasp_validity'] = {
                'x_axis': validity_info.get('valid_x', []),
                'y_axis': validity_info.get('valid_y', [])
            }
            filtered_points_with_metadata.append(enhanced_gp)

        grasp_data['grasp_points'] = filtered_points_with_metadata
        grasp_data['total_points'] = len(filtered_points_with_metadata)
        grasp_data['filter_applied'] = True
        grasp_data['filter_params'] = filter_result['filter_params']

        # Save filtered grasp points
        with open(grasp_points_json, 'w') as f:
            json.dump(grasp_data, f, indent=2)

        # Generate filtered visualization
        filtered_viz_base64 = _generate_filtered_visualization(
            object_name, marker_id, regions, filter_result, rendered_image_path
        )

        return {
            "success": True,
            "object_name": object_name,
            "marker_id": marker_id,
            "total_original": filter_result['total_original'],
            "total_filtered": filter_result['total_filtered'],
            "filter_results": filter_result['results'],
            "filter_params": filter_result['filter_params'],
            "filtered_visualization": f"data:image/png;base64,{filtered_viz_base64}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _generate_filtered_visualization(
    object_name: str,
    marker_id: int,
    regions: list,
    filter_result: dict,
    rendered_image_path: Path
) -> str:
    """Generate visualization showing only filtered grasp points."""
    # Get valid regions
    valid_grasp_ids = {r['grasp_id'] for r in filter_result['results'] if r['is_valid']}
    filtered_regions = [r for r in regions if r['id'] in valid_grasp_ids]

    # Load rendered image
    rendered_image = cv2.imread(str(rendered_image_path))

    # Load mask
    mask_path = OUTPUTS_DIR / "masks" / f"{object_name}_marker{marker_id}_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)

    # Create filtered visualization
    filtered_viz_output = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_grasp_points_filtered.png"
    visualize_center_points_only(str(rendered_image_path), mask, filtered_regions, str(filtered_viz_output))

    # Convert to base64
    with open(filtered_viz_output, 'rb') as f:
        viz_data = f.read()
        return base64.b64encode(viz_data).decode('utf-8')
