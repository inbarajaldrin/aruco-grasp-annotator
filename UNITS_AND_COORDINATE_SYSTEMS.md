# Units and Coordinate Systems Reference

This document describes the units and coordinate systems used throughout the ArUco Grasp Annotator project.

## Unit Conventions

### Summary
All **exported data** (wireframes, ArUco annotations, grasp points) use **METERS** as the standard unit, even though the original CAD models may be in centimeters.

### Detailed Breakdown

| Data Type | Storage Format | Units | Example Values |
|-----------|---------------|-------|----------------|
| **Original CAD Models** (`.obj` files) | Raw mesh vertices | **Centimeters** | `[1.05, -3.5, -1.05]` |
| **Wireframe JSON** | Exported vertices/edges | **Meters** | `[0.0105, -0.035, -0.0105]` |
| **ArUco Annotations JSON** | Marker positions/sizes | **Meters** | `position: -0.0546, size: 0.021` |
| **Grasp Points JSON** | 3D grasp coordinates | **Meters** | `x: -0.017804, y: -0.049506` |
| **Assembly App Visualization** | Three.js scene coordinates | **Meters** | Sphere radius: `0.003` (3mm) |

### Conversion Factor
**Original CAD (cm) → Exported Data (m)**: Divide by **100**

Example:
- CAD vertex: `[1.05, -3.5, -1.05]` cm
- Wireframe vertex: `[0.0105, -0.035, -0.0105]` m

## Coordinate Systems

### 1. CAD Model Frame
- **Origin**: Center of the CAD object's bounding box
- **Axes**: Standard 3D Cartesian (X, Y, Z)
- **Units**: Centimeters (in raw `.obj` files)

### 2. ArUco Marker Frame
- **Origin**: Center of the ArUco marker surface
- **Z-axis**: Normal to the marker surface (pointing outward)
- **Units**: Meters (in annotation files)
- **Usage**: Grasp points are defined relative to each marker's frame

### 3. Camera Frame (for rendering)
- **Origin**: Camera position (top-down view)
- **Z-axis**: Points downward (toward the object)
- **Projection**: Orthographic (no perspective distortion)
- **Units**: Meters

### 4. World/Scene Frame (Assembly App)
- **Origin**: Scene center
- **Axes**: Three.js standard (Y-up)
- **Units**: Meters
- **Purpose**: All objects (wireframes, markers, grasp points) are positioned in this frame

## Grasp Points Pipeline

### Data Flow and Transformations

```
1. CAD Model (cm)
   ↓ [Load mesh]
2. Transformed Mesh (cm) - marker facing up
   ↓ [Orthographic render]
3. 2D Image (pixels) + Depth Map (cm)
   ↓ [Grasp detection]
4. 2D Grasp Points (pixels)
   ↓ [2D-to-3D mapping using depth]
5. 3D Grasp Points in Marker Frame (cm)
   ↓ [Transform to all markers]
6. 3D Grasp Points for All Markers (cm)
   ↓ [÷100 conversion]
7. Final Grasp JSON (m) ✓
```

### Critical Conversion Point

**File**: `src/grasp_points/annotate_grasp_to_all_markers.py`

```python
# Convert from centimeters to meters to match wireframe/ArUco scale
grasp_points_in_marker.append({
    "position": {
        "x": float(transformed_point[0] / 100.0),  # cm → m
        "y": float(transformed_point[1] / 100.0),  # cm → m
        "z": float(transformed_point[2] / 100.0)   # cm → m
    }
})
```

## Visualization Scale Reference

### Assembly App (Three.js)

All measurements in **meters**:

- **Grasp Point Spheres**: `0.003 m` radius (3mm diameter)
- **Approach Arrows**: `0.01 m` length (1cm)
- **ArUco Markers**: Actual size from annotations (e.g., `0.021 m` = 2.1cm)
- **Wireframe Lines**: Direct vertex positions in meters

### Typical Object Dimensions (Example: Fork)

- **Bounding Box**: `[0.1092, 0.07, 0.021]` m = `[10.92, 7.0, 2.1]` cm
- **Length**: ~11 cm
- **Width**: ~7 cm  
- **Thickness**: ~2.1 cm

## Important Notes

1. **Consistency is Key**: All exported/processed data must be in meters for the assembly app to work correctly.

2. **Original CAD Files**: Can be in any unit, but are typically in centimeters for this project.

3. **Focal Length Calculation**: For orthographic projection rendering, the focal length is calculated as:
   ```python
   focal_length = image_size / max_dimension
   ```
   Where `max_dimension` is in the same units as the CAD model (cm).

4. **Depth Values**: In the rendering pipeline, depth values are in centimeters (matching the transformed mesh units) and must be converted to meters in the final output.

## Verification Checklist

When adding new objects or grasp points, verify:

- [ ] Wireframe vertices are in meters (values typically 0.001 to 0.5)
- [ ] ArUco marker positions are in meters
- [ ] ArUco marker sizes are in meters (typically 0.02 to 0.03)
- [ ] Grasp point positions are in meters
- [ ] All data aligns correctly in the assembly app visualization

## Troubleshooting

### Objects appear too large/small in visualization
- Check that all data (wireframe, markers, grasp points) use the same units
- Verify conversion factor (÷100) is applied where needed

### Grasp points don't align with the object
- Ensure grasp points are in meters, not centimeters
- Verify the coordinate transformation pipeline
- Check that the marker frame transformations are correct

### Grasp points appear inverted
- **Root cause**: Coordinate system mismatch between CAD and Three.js/wireframe
- **Solution**: Coordinate flip (`-X, -Y, +Z`) is applied during export in `annotate_grasp_to_all_markers.py`
- The wireframe exporter already applies this flip, so grasp points must match
- Do NOT apply additional flips in visualization code

### Markers floating away from the object
- Verify ArUco annotation positions are in meters
- Check that the CAD center is correctly calculated

---

**Last Updated**: October 6, 2025  
**Related Files**: 
- `src/grasp_points/annotate_grasp_to_all_markers.py` - Grasp point unit conversion
- `src/grasp_points/cad_to_image_renderer.py` - Rendering and intrinsics
- `src/assembly_app/app.py` - Visualization scale settings

