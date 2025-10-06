# Grasp Points Pipeline

Automatically generate 3D grasp points from CAD models with ArUco markers.

## Quick Start

```bash
cd src/grasp_points
python cad_to_grasp_pipeline.py --object fork_orange_scaled70 --marker-id 5
```

## What It Does

1. **Renders** CAD model with specified ArUco marker facing up (orthographic view)
2. **Detects** grasp regions using adaptive region splitting
3. **Maps** 2D points to 3D coordinates relative to the ArUco marker
4. **Exports** grasp points as JSON for robotics applications

## Usage

```bash
python cad_to_grasp_pipeline.py --object <object_name> --marker-id <id>
```

### Examples

```bash
# Generate grasp points for base with marker 24
python cad_to_grasp_pipeline.py --object base_scaled70 --marker-id 24

# Generate grasp points for fork with marker 5
python cad_to_grasp_pipeline.py --object fork_orange_scaled70 --marker-id 5
```

## Output Files

All files saved in `outputs/` folder:

- `{object}_marker{id}_topdown.png` - Orthographic render
- `{object}_marker{id}_grasp_points_2d.png` - Detected regions visualization
- `{object}_marker{id}_grasp_points_3d.json` - **Final 3D grasp points** ⭐

## Requirements

- Object must be annotated with ArUco markers (use main annotator tool)
- CAD file in `../../data/models/`
- ArUco annotation in `../../data/aruco/`

## Key Features

✅ True orthographic projection (no perspective distortion)  
✅ Preserves object aspect ratio  
✅ All coordinates relative to ArUco marker  
✅ Works with any marker orientation  

## Transform Grasp Points to All Markers

After generating grasp points for one marker, transform them to all markers:

```bash
# Auto-detects object thickness from ArUco annotations
python annotate_grasp_to_all_markers.py --object fork_orange_scaled70 --source-marker-id 5

# Or manually specify thickness (in meters)
python annotate_grasp_to_all_markers.py --object fork_orange_scaled70 --source-marker-id 5 --object-thickness 0.021
```

This creates a comprehensive JSON file in `../../data/grasp/` with:
- Wireframe data
- All ArUco markers
- Grasp points relative to each marker

## Files

- `cad_to_grasp_pipeline.py` - Main script (run this)
- `annotate_grasp_to_all_markers.py` - Transform to all markers
- `cad_to_image_renderer.py` - Orthographic rendering
- `point_mapper_2d_to_3d.py` - 2D to 3D mapping
- `adaptive_region_center_points.py` - Region detection algorithm

