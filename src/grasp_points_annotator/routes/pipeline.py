"""Pipeline routes for Grasp Points Annotator (Step 1 and Step 2)."""

import base64
import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Body, HTTPException

from shared.fastapi_utils import get_data_dir

from ..core.annotation_transformer import annotate_grasp_points_to_all_markers
from ..core.pipeline import CADToGraspPipeline
from ..utils.region_detector import visualize_center_points_only

router = APIRouter(prefix="/api/pipeline")

# Directory paths
DATA_DIR = get_data_dir(Path(__file__).parent.parent / "app.py")
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


@router.post("/step1")
async def run_step1(data: dict[str, Any] = Body(...)):
    """
    Run Step 1: CAD to Grasp Points.
    Renders top-down view and detects grasp points.
    """
    try:
        object_name = data.get("object_name")
        marker_id = data.get("marker_id")
        camera_distance = data.get("camera_distance", 0.5)
        min_area_threshold = data.get("min_area_threshold", 1000)
        use_mtl_color = data.get("use_mtl_color", False)

        if not object_name or marker_id is None:
            raise HTTPException(status_code=400, detail="object_name and marker_id are required")

        # Initialize pipeline with absolute paths
        pipeline = CADToGraspPipeline(
            data_dir=str(DATA_DIR.resolve()),
            outputs_dir=str(OUTPUTS_DIR.resolve())
        )

        # Run pipeline
        result = pipeline.run(
            object_name=object_name,
            marker_id=marker_id,
            camera_distance=camera_distance,
            min_area_threshold=min_area_threshold,
            use_mtl_color=use_mtl_color
        )

        # Convert image to base64 for frontend
        image = result['render_data']['image']
        # Convert RGBA to BGRA for OpenCV encoding
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Get visualization image
        viz_path = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_grasp_points_2d.png"
        viz_base64 = None
        if viz_path.exists():
            with open(viz_path, 'rb') as f:
                viz_data = f.read()
                viz_base64 = base64.b64encode(viz_data).decode('utf-8')

        return {
            "success": True,
            "object_name": object_name,
            "marker_id": marker_id,
            "points_2d_count": len(result['points_2d']),
            "points_3d_count": len(result['points_3d']),
            "rendered_image": f"data:image/png;base64,{image_base64}",
            "visualization_image": f"data:image/png;base64,{viz_base64}" if viz_base64 else None,
            "output_json": str(result['output_json']),
            "points_3d": [[float(p[0]), float(p[1]), float(p[2])] for p in result['points_3d']]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step2")
async def run_step2(data: dict[str, Any] = Body(...)):
    """
    Run Step 2: Transform to All Markers.
    Transforms grasp points from source marker to all markers.
    """
    try:
        object_name = data.get("object_name")
        source_marker_id = data.get("source_marker_id")
        object_thickness = data.get("object_thickness", None)

        if not object_name or source_marker_id is None:
            raise HTTPException(status_code=400, detail="object_name and source_marker_id are required")

        # Find the grasp points JSON from step 1
        grasp_points_json = OUTPUTS_DIR / f"{object_name}_marker{source_marker_id}_grasp_points_3d.json"
        if not grasp_points_json.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Grasp points file not found. Please run Step 1 first: {grasp_points_json}"
            )

        # Run transformation with absolute paths
        output_file = annotate_grasp_points_to_all_markers(
            object_name=object_name,
            source_marker_id=source_marker_id,
            grasp_points_json_path=str(grasp_points_json.resolve()),
            object_thickness=object_thickness,
            data_dir=str(DATA_DIR.resolve())
        )

        # Load the output to return summary
        with open(output_file, 'r') as f:
            output_data = json.load(f)

        # Generate visualization with renumbered IDs
        renumbered_viz_base64 = _generate_renumbered_visualization(
            object_name, source_marker_id, grasp_points_json
        )

        return {
            "success": True,
            "object_name": object_name,
            "source_marker_id": source_marker_id,
            "output_file": str(output_file),
            "total_grasp_points": output_data.get("total_grasp_points", 0),
            "total_markers": len(output_data.get("markers", [])),
            "grasp_points": output_data.get("grasp_points", []),
            "renumbered_visualization": f"data:image/png;base64,{renumbered_viz_base64}" if renumbered_viz_base64 else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_renumbered_visualization(
    object_name: str,
    source_marker_id: int,
    grasp_points_json: Path
) -> str | None:
    """Generate visualization with renumbered grasp point IDs."""
    try:
        rendered_image_path = OUTPUTS_DIR / f"{object_name}_marker{source_marker_id}_topdown.png"
        if not rendered_image_path.exists():
            return None

        # Load the mask
        mask_path = OUTPUTS_DIR / "masks" / f"{object_name}_marker{source_marker_id}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            rendered_image = cv2.imread(str(rendered_image_path))
            mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)

        # Re-detect regions to get their pixel coordinates
        pipeline = CADToGraspPipeline(
            data_dir=str(DATA_DIR.resolve()),
            outputs_dir=str(OUTPUTS_DIR.resolve())
        )

        _, regions, _ = pipeline.detect_grasp_points_2d(
            rendered_image_path, object_name, source_marker_id, min_area_threshold=1000
        )

        # Load step 1 data to get filtered IDs
        with open(grasp_points_json, 'r') as f:
            step1_data = json.load(f)

        filtered_ids = {gp['id'] for gp in step1_data.get('grasp_points', [])}

        # Filter and renumber regions
        renumbered_regions = []
        for new_id, region in enumerate((r for r in regions if r['id'] in filtered_ids), 1):
            renumbered_region = region.copy()
            renumbered_region['id'] = new_id
            renumbered_regions.append(renumbered_region)

        if not renumbered_regions:
            return None

        # Generate renumbered visualization
        renumbered_viz_output = OUTPUTS_DIR / f"{object_name}_marker{source_marker_id}_grasp_points_renumbered.png"
        visualize_center_points_only(str(rendered_image_path), mask, renumbered_regions, str(renumbered_viz_output))

        # Convert to base64
        with open(renumbered_viz_output, 'rb') as f:
            viz_data = f.read()
            return base64.b64encode(viz_data).decode('utf-8')

    except Exception as viz_error:
        print(f"Warning: Could not generate renumbered visualization: {viz_error}")
        import traceback
        traceback.print_exc()
        return None
