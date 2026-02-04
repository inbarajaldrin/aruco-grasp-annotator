"""Marker image routes for Assembly App."""

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from shared.aruco_utils import generate_aruco_marker

router = APIRouter(prefix="/api")


@router.get("/marker-image")
async def get_marker_image(
    dictionary: str = Query("DICT_4X4_50", description="ArUco dictionary name"),
    marker_id: int = Query(..., description="ArUco marker ID"),
    size: int = Query(512, description="Image size in pixels"),
):
    """Generate ArUco marker image and return as PNG."""
    try:
        marker_img = generate_aruco_marker(dictionary, marker_id, size)
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
