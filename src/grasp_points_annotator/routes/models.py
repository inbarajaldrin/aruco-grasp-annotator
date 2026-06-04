"""
Typed request bodies for Grasp Points Annotator endpoints.

Mirror the prior loose ``dict`` bodies field-for-field so OpenAPI describes the inputs
(groundwork for MCP tool schemas). ``object_name``/``marker_id`` stay Optional so the
handlers' own "required" 400 checks are preserved exactly; unknown keys are ignored.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Step1Request(BaseModel):
    """Body for POST /api/pipeline/step1 (CAD -> grasp points)."""

    model_config = ConfigDict(extra="ignore")
    object_name: Optional[str] = Field(None, description="Object/model name")
    marker_id: Optional[int] = Field(None, description="Source ArUco marker id")
    camera_distance: float = Field(0.5, description="Top-down camera distance (m)")
    min_area_threshold: int = Field(1000, description="Min region area (px) to keep")
    use_mtl_color: bool = Field(False, description="Use the model's .mtl colors when rendering")


class Step2Request(BaseModel):
    """Body for POST /api/pipeline/step2 (transform to all markers)."""

    model_config = ConfigDict(extra="ignore")
    object_name: Optional[str] = Field(None, description="Object/model name")
    source_marker_id: Optional[int] = Field(None, description="Marker the grasp points came from")
    object_thickness: Optional[float] = Field(
        None, description="Object thickness (m); inferred if omitted"
    )


class FilterRequest(BaseModel):
    """Body for POST /api/filter (gripper-constraint filtering)."""

    model_config = ConfigDict(extra="ignore")
    object_name: Optional[str] = Field(None, description="Object/model name")
    marker_id: Optional[int] = Field(None, description="Source ArUco marker id")
    gripper_max_width_mm: float = Field(100.0, description="Max gripper opening (mm)")
    grasp_clearance_mm: float = Field(14.0, description="Required clearance around grasp (mm)")
    gripper_tip_thickness_mm: float = Field(20.0, description="Gripper tip thickness (mm)")
    max_gap_px: int = Field(20, description="Max gap to bridge between regions (px)")
    symmetry_tolerance_mm: float = Field(10.0, description="Symmetry matching tolerance (mm)")
    check_x_axis: bool = Field(True, description="Check grasps along the X axis")
    check_y_axis: bool = Field(True, description="Check grasps along the Y axis")
