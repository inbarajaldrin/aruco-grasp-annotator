#!/usr/bin/env python3
"""
Grasp Points Annotation Web Application
FastAPI-based web application for generating grasp points from CAD models
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import socket
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
import base64
import io
import cv2

from .core.pipeline import CADToGraspPipeline
from .core.annotation_transformer import annotate_grasp_points_to_all_markers
from .utils.grasp_filter import GraspFilter

app = FastAPI(
    title="Grasp Points Annotation App",
    description="Interactive web application for generating grasp points from CAD models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory path - automatically resolve relative to project root
# This works whether running from project root or from the app directory
_app_dir = Path(__file__).parent
_project_root = _app_dir.parent.parent
DATA_DIR = _project_root / "data"
OUTPUTS_DIR = _app_dir / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Debug: Print paths on startup
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Data directory exists: {DATA_DIR.exists()}")
print(f"📁 Outputs directory: {OUTPUTS_DIR}")


def get_available_objects():
    """Dynamically discover objects from models directory."""
    models_dir = DATA_DIR / "models"
    aruco_dir = DATA_DIR / "aruco"
    objects = []
    
    # First, check if directories exist
    if not models_dir.exists():
        print(f"Warning: Models directory not found: {models_dir}")
        return []
    
    if not aruco_dir.exists():
        print(f"Warning: ArUco directory not found: {aruco_dir}")
        return []
    
    # Get all ArUco annotation files
    aruco_files = list(aruco_dir.glob("*_aruco.json"))
    
    for aruco_file in aruco_files:
        # Extract object name from ArUco file (e.g., "base_scaled70_aruco.json" -> "base_scaled70")
        object_name = aruco_file.stem.replace("_aruco", "")
        
        # Check if corresponding CAD model exists (try multiple extensions)
        model_found = False
        for ext in ['.obj', '.stl', '.ply']:
            model_file = models_dir / f"{object_name}{ext}"
            if model_file.exists():
                model_found = True
                break
        
        if model_found:
            objects.append(object_name)
        else:
            print(f"Warning: No CAD model found for {object_name} (checked .obj, .stl, .ply)")
    
    return sorted(objects)


def get_markers_for_object(object_name: str) -> List[int]:
    """Get list of marker IDs for an object."""
    aruco_file = DATA_DIR / "aruco" / f"{object_name}_aruco.json"
    if not aruco_file.exists():
        return []
    
    with open(aruco_file, 'r') as f:
        data = json.load(f)
    
    marker_ids = [marker['aruco_id'] for marker in data.get('markers', [])]
    return sorted(marker_ids)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application interface."""
    html_content = Path(__file__).parent / "templates" / "index.html"
    if html_content.exists():
        return html_content.read_text()
    
    # Fallback: return basic HTML
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grasp Points Annotation App</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>Grasp Points Annotation App</h1>
        <p>Template file not found. Please check templates/index.html</p>
    </body>
    </html>
    """


@app.get("/api/objects")
async def get_objects():
    """Get list of available objects."""
    try:
        objects = get_available_objects()
        print(f"📦 Found {len(objects)} objects: {objects}")
        return {"objects": objects}
    except Exception as e:
        print(f"❌ Error getting objects: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error discovering objects: {str(e)}")


@app.get("/api/markers/{object_name}")
async def get_markers(object_name: str):
    """Get markers for a specific object."""
    marker_ids = get_markers_for_object(object_name)
    return {"object_name": object_name, "markers": marker_ids}


@app.post("/api/pipeline/step1")
async def run_step1(data: Dict[str, Any] = Body(...)):
    """
    Run Step 1: CAD to Grasp Points
    Renders top-down view and detects grasp points.
    """
    try:
        object_name = data.get("object_name")
        marker_id = data.get("marker_id")
        camera_distance = data.get("camera_distance", 0.5)
        min_area_threshold = data.get("min_area_threshold", 1000)
        
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
            min_area_threshold=min_area_threshold
        )
        
        # Convert image to base64 for frontend
        import cv2
        import numpy as np
        
        image = result['render_data']['image']
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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


@app.post("/api/pipeline/step2")
async def run_step2(data: Dict[str, Any] = Body(...)):
    """
    Run Step 2: Transform to All Markers
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
        
        return {
            "success": True,
            "object_name": object_name,
            "source_marker_id": source_marker_id,
            "output_file": str(output_file),
            "total_grasp_points": output_data.get("total_grasp_points", 0),
            "total_markers": len(output_data.get("markers", [])),
            "grasp_points": output_data.get("grasp_points", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{object_name}")
async def get_results(object_name: str):
    """Get results for an object."""
    result_file = DATA_DIR / "grasp" / f"{object_name}_grasp_points_all_markers.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    return data


@app.get("/api/download/{object_name}")
async def download_results(object_name: str):
    """Download the final results JSON file."""
    result_file = DATA_DIR / "grasp_points" / f"{object_name}_grasp_points.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    return FileResponse(
        path=str(result_file),
        filename=f"{object_name}_grasp_points.json",
        media_type="application/json"
    )


@app.post("/api/filter")
async def filter_grasp_points(data: Dict[str, Any] = Body(...)):
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

        # Need to run region detection to get regions for filtering
        pipeline = CADToGraspPipeline(
            data_dir=str(DATA_DIR.resolve()),
            outputs_dir=str(OUTPUTS_DIR.resolve())
        )

        # Get the rendered image path
        rendered_image_path = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_topdown.png"
        if not rendered_image_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Rendered image not found. Please run Step 1 first."
            )

        # Detect regions
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

        # Create a grasp_data structure with wireframe for the filter
        filter_grasp_data = {
            'grasp_points': grasp_data.get('grasp_points', []),
            'wireframe': {
                'vertices': wireframe_data['vertices']
            }
        }

        # Initialize filter with custom parameters
        grasp_filter = GraspFilter(
            gripper_max_width_mm=gripper_max_width_mm,
            gripper_half_open_width_mm=gripper_half_open_width_mm,
            gripper_tip_thickness_mm=gripper_tip_thickness_mm,
            max_gap_px=max_gap_px,
            symmetry_tolerance_mm=symmetry_tolerance_mm
        )

        # Run filtering
        filter_result = grasp_filter.filter_grasp_points(
            filter_grasp_data,
            regions,
            check_x_axis=check_x_axis,
            check_y_axis=check_y_axis
        )

        # Create a mapping of grasp_id to validity info
        validity_map = {r['grasp_id']: r for r in filter_result['results']}

        # Update the grasp data with filtered points and add validity metadata
        filtered_points_with_metadata = []
        for gp in filter_result['filtered_grasp_points']:
            grasp_id = gp['id']
            validity_info = validity_map.get(grasp_id, {})

            # Add validity metadata to the grasp point
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

        # Save the filtered grasp points back to the file
        with open(grasp_points_json, 'w') as f:
            json.dump(grasp_data, f, indent=2)

        # Generate updated visualization with only filtered grasp points
        from .utils.region_detector import visualize_center_points_only

        # Get only the valid regions (those that passed the filter)
        valid_grasp_ids = {r['grasp_id'] for r in filter_result['results'] if r['is_valid']}
        filtered_regions = [r for r in regions if r['id'] in valid_grasp_ids]

        # Load the rendered image
        rendered_image = cv2.imread(str(rendered_image_path))

        # Load the mask
        mask_path = OUTPUTS_DIR / "masks" / f"{object_name}_marker{marker_id}_mask.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create a simple mask from the rendered image if not available
            mask = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)

        # Create filtered visualization
        filtered_viz_output = OUTPUTS_DIR / f"{object_name}_marker{marker_id}_grasp_points_filtered.png"
        visualize_center_points_only(str(rendered_image_path), mask, filtered_regions, str(filtered_viz_output))

        # Convert filtered visualization to base64
        with open(filtered_viz_output, 'rb') as f:
            viz_data = f.read()
            filtered_viz_base64 = base64.b64encode(viz_data).decode('utf-8')

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


def find_available_port(start_port=8002, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port starting from {start_port}")

def main():
    """Main entry point for the grasp points annotation application."""
    # Find an available port starting from 8002
    port = find_available_port(8002)
    
    print("🚀 Starting Grasp Points Annotation Web App...")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("🎯 Features:")
    print("   • Load CAD models and ArUco annotations")
    print("   • Generate grasp points from top-down views")
    print("   • Transform grasp points to all markers")
    print("   • Export results for robotics pipelines")
    print("🎮 Usage:")
    print("   • Select object and marker")
    print("   • Run Step 1: Generate grasp points")
    print("   • Run Step 2: Transform to all markers")
    print("   • Download final results")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

