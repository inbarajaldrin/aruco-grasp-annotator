"""Export and import routes for ArUco Annotator."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from scipy.spatial.transform import Rotation

from shared.fastapi_utils import get_data_dir as _get_data_dir

from ..models.marker import MarkerData
from ..models.session import session_state
from ..utils.aruco_utils import ArUcoGenerator

router = APIRouter(prefix="/api")

# Initialize ArUco generator
aruco_generator = ArUcoGenerator()


def get_data_dir() -> Path:
    """Get the data directory path."""
    return _get_data_dir(Path(__file__).parent.parent / "app.py")


@router.get("/marker-image")
async def get_marker_image(
    dictionary: str = Query(..., description="ArUco dictionary name"),
    marker_id: int = Query(..., description="ArUco marker ID"),
    size: int = Query(512, description="Image size in pixels"),
):
    """Generate ArUco marker image and return as PNG."""
    try:
        marker_img = aruco_generator.generate_marker(dictionary, marker_id, size)
        _, buffer = cv2.imencode(".png", marker_img)
        img_bytes = buffer.tobytes()

        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating marker image: {str(e)}"
        )


@router.get("/export")
async def export_annotations():
    """Export annotations to JSON file."""
    if len(session_state.markers) == 0:
        raise HTTPException(status_code=400, detail="No markers to export")

    current_file = session_state.current_file
    if not current_file:
        raise HTTPException(
            status_code=400,
            detail="No CAD file name available. Please load a CAD model first.",
        )

    first_marker = list(session_state.markers.values())[0]
    cad_info = session_state.cad_object_info

    markers_list = []
    for internal_id, marker in session_state.markers.items():
        T_object_to_marker = marker.get_T_object_to_marker(cad_info["center"])

        markers_list.append(
            {
                "aruco_id": marker.aruco_id,
                "face_type": marker.face_type,
                "surface_normal": marker.face_normal.tolist(),
                "T_object_to_marker": T_object_to_marker,
            }
        )

    export_data = {
        "exported_at": datetime.now().isoformat(),
        "model_file": session_state.current_file,
        "total_markers": len(markers_list),
        "aruco_dictionary": first_marker.dictionary,
        "size": first_marker.size,
        "border_width": first_marker.border_width,
        "cad_object_info": {
            "center": cad_info["center"],
            "dimensions": cad_info["dimensions"],
            "position": cad_info.get("position", [0.0, 0.0, 0.0]),
            "rotation": cad_info.get(
                "rotation",
                {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            ),
        },
        "markers": markers_list,
        "notes": "T_object_to_marker is the complete transform from object center to marker.",
    }

    json_str = json.dumps(export_data, indent=2)
    object_name = Path(current_file).stem

    data_dir = get_data_dir()
    aruco_dir = data_dir / "aruco"
    aruco_dir.mkdir(parents=True, exist_ok=True)
    aruco_file = aruco_dir / f"{object_name}_aruco.json"

    try:
        with open(aruco_file, "w") as f:
            f.write(json_str)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving annotations to {aruco_file}: {str(e)}",
        )

    aruco_filename = f"{object_name}_aruco.json"
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={aruco_filename}"},
    )


@router.get("/export-wireframe")
async def export_wireframe():
    """Export wireframe data from the loaded mesh."""
    if session_state.mesh is None:
        raise HTTPException(
            status_code=400, detail="No mesh loaded. Please load a CAD model first."
        )

    current_file = session_state.current_file
    if not current_file:
        raise HTTPException(
            status_code=400,
            detail="No CAD file name available. Please load a CAD model first.",
        )

    try:
        mesh = session_state.mesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        edges = []
        for triangle in triangles:
            for i in range(3):
                v1, v2 = triangle[i], triangle[(i + 1) % 3]
                edge = tuple(sorted([int(v1), int(v2)]))
                edges.append(edge)

        unique_edges = []
        seen = set()
        for edge in edges:
            if edge not in seen:
                unique_edges.append(edge)
                seen.add(edge)

        mesh_info = {
            "num_vertices": len(vertices),
            "num_edges": len(unique_edges),
            "num_triangles": len(triangles),
            "bounding_box": {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
                "center": vertices.mean(axis=0).tolist(),
                "size": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
            },
            "has_normals": mesh.has_vertex_normals(),
            "has_colors": mesh.has_vertex_colors(),
            "is_watertight": mesh.is_watertight(),
            "is_orientable": mesh.is_orientable(),
        }

        wireframe_data = {
            "mesh_info": mesh_info,
            "vertices": vertices.tolist(),
            "edges": [[int(edge[0]), int(edge[1])] for edge in unique_edges],
            "format": "vector_relation",
            "description": "Wireframe data with vertices and edge connections",
        }

        json_str = json.dumps(wireframe_data, indent=2)
        object_name = Path(current_file).stem

        data_dir = get_data_dir()
        wireframe_dir = data_dir / "wireframe"
        wireframe_dir.mkdir(parents=True, exist_ok=True)
        wireframe_file = wireframe_dir / f"{object_name}_wireframe.json"

        try:
            with open(wireframe_file, "w") as f:
                f.write(json_str)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error saving wireframe to {wireframe_file}: {str(e)}",
            )

        wireframe_filename = f"{object_name}_wireframe.json"
        return Response(
            content=json_str,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={wireframe_filename}"
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to export wireframe: {str(e)}"
        )


@router.post("/import")
async def import_annotations(file: UploadFile = File(...)):
    """Import annotations from uploaded JSON file."""
    if session_state.cad_object_info is None:
        raise HTTPException(
            status_code=400,
            detail="No CAD model loaded. Please load a CAD model first.",
        )

    try:
        content = await file.read()
        data = json.loads(content)
        return await _process_imported_annotations(data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error importing annotations: {str(e)}"
        )


@router.post("/import-auto")
async def import_annotations_auto():
    """Automatically import annotations from data folder."""
    if session_state.cad_object_info is None:
        raise HTTPException(
            status_code=400,
            detail="No CAD model loaded. Please load a CAD model first.",
        )

    current_file = session_state.current_file
    if not current_file:
        raise HTTPException(
            status_code=400,
            detail="No CAD file name available. Please load a CAD model first.",
        )

    object_name = Path(current_file).stem
    data_dir = get_data_dir()
    aruco_file = data_dir / "aruco" / f"{object_name}_aruco.json"

    if not aruco_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Annotation file not found in data folder: {aruco_file}",
        )

    try:
        with open(aruco_file, "r") as f:
            data = json.load(f)
        return await _process_imported_annotations(data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading annotations from {aruco_file}: {str(e)}",
        )


async def _process_imported_annotations(data: dict) -> JSONResponse:
    """Process imported annotation data and update session state."""
    session_state.markers = {}
    session_state.next_marker_id = 0

    cad_info = data.get("cad_object_info", {})
    cad_center = np.array(cad_info.get("center", [0, 0, 0]))

    if session_state.cad_object_info is not None:
        imported_cad_info = data.get("cad_object_info", {})

        if "position" in imported_cad_info:
            session_state.cad_object_info["position"] = imported_cad_info["position"]

        if "rotation" in imported_cad_info:
            session_state.cad_object_info["rotation"] = imported_cad_info["rotation"]
        else:
            session_state.cad_object_info["rotation"] = {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }

    try:
        for marker_data in data.get("markers", []):
            aruco_id = marker_data["aruco_id"]

            T_object_to_marker = marker_data.get("T_object_to_marker")
            if T_object_to_marker is None:
                T_marker_to_object = marker_data.get("T_marker_to_object", {})
                if T_marker_to_object:
                    rel_pos = T_marker_to_object.get("position", {})
                    rel_rot = T_marker_to_object.get("rotation", {})
                    relative_pos_marker_frame = np.array(
                        [rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)]
                    )
                    R_marker_to_object_euler = (
                        rel_rot.get("roll", 0),
                        rel_rot.get("pitch", 0),
                        rel_rot.get("yaw", 0),
                    )
                    R_marker_to_object = Rotation.from_euler(
                        "xyz", R_marker_to_object_euler
                    ).as_matrix()
                    relative_pos_object_frame = (
                        R_marker_to_object @ relative_pos_marker_frame
                    )
                    position_local = tuple(cad_center - relative_pos_object_frame)
                    R_object_to_marker = R_marker_to_object.T
                else:
                    position_local = tuple(cad_center)
                    R_object_to_marker = np.eye(3)
            else:
                rel_pos = T_object_to_marker.get("position", {})
                rel_rot = T_object_to_marker.get("rotation", {})

                relative_pos_object_frame = np.array(
                    [rel_pos.get("x", 0), rel_pos.get("y", 0), rel_pos.get("z", 0)]
                )

                R_object_to_marker_euler = (
                    rel_rot.get("roll", 0),
                    rel_rot.get("pitch", 0),
                    rel_rot.get("yaw", 0),
                )
                R_object_to_marker = Rotation.from_euler(
                    "xyz", R_object_to_marker_euler
                ).as_matrix()

                position_local = tuple(cad_center + relative_pos_object_frame)

            surface_normal = tuple(marker_data.get("surface_normal", [0, 0, 1]))
            face_type = marker_data.get("face_type", "unknown")

            marker = MarkerData(
                aruco_id=aruco_id,
                dictionary=data.get("aruco_dictionary", "DICT_4X4_50"),
                size=data.get("size", 0.021),
                border_width=data.get("border_width", 0.05),
                position=position_local,
                face_normal=surface_normal,
                face_type=face_type,
            )

            imported_translation = marker_data.get("translation_offset")
            if imported_translation is not None:
                x_offset = imported_translation.get("x", 0.0)
                y_offset = imported_translation.get("y", 0.0)
                marker.in_plane_translation = np.array([x_offset, y_offset])

                R_base = marker.base_rotation_matrix
                marker_x_axis = R_base[:, 0]
                marker_y_axis = R_base[:, 1]
                translation_vector = x_offset * marker_x_axis + y_offset * marker_y_axis
                marker.initial_position = marker.position - translation_vector
            else:
                marker.in_plane_translation = np.array([0.0, 0.0])
                marker.initial_position = marker.position.copy()

            in_plane_rotation_deg = marker_data.get("in_plane_rotation_deg")
            if in_plane_rotation_deg is None:
                R_full = R_object_to_marker
                R_base = marker.base_rotation_matrix
                R_inplane = (R_full @ R_base).T
                rot_inplane = Rotation.from_matrix(R_inplane)
                euler_inplane = rot_inplane.as_euler("xyz")
                in_plane_rotation_deg = np.rad2deg(euler_inplane[2])
            else:
                in_plane_rotation_deg = float(in_plane_rotation_deg)

            marker.set_in_plane_rotation(in_plane_rotation_deg)

            internal_id = session_state.next_marker_id
            session_state.markers[internal_id] = marker
            session_state.next_marker_id = max(
                session_state.next_marker_id, internal_id + 1
            )

        return JSONResponse(
            {"success": True, "imported": len(session_state.markers)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
