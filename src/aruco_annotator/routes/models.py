"""
Typed request bodies for ArUco Annotator endpoints.

These mirror the loose ``dict`` bodies the handlers previously accepted, field-for-field,
so OpenAPI (and any downstream MCP tool derived from it) describes the inputs precisely.
Defaults match the prior ``body.get(key, default)`` behavior exactly. Unknown keys are
ignored (Pydantic v2 default), matching handlers that only read known keys.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Vec3(BaseModel):
    """A 3D point/vector. Components default to 0."""

    model_config = ConfigDict(extra="ignore")
    x: float = Field(0.0, description="X component")
    y: float = Field(0.0, description="Y component")
    z: float = Field(0.0, description="Z component")


class NormalVec3(BaseModel):
    """A surface normal. Defaults to +Z (0, 0, 1), matching the prior dict default."""

    model_config = ConfigDict(extra="ignore")
    x: float = Field(0.0, description="X component")
    y: float = Field(0.0, description="Y component")
    z: float = Field(1.0, description="Z component (defaults to 1)")


class RPY(BaseModel):
    """Euler angles in degrees. Components default to 0."""

    model_config = ConfigDict(extra="ignore")
    roll: float = Field(0.0, description="Roll in degrees")
    pitch: float = Field(0.0, description="Pitch in degrees")
    yaw: float = Field(0.0, description="Yaw in degrees")


class AddMarkerConfig(BaseModel):
    """Body for POST /api/add-marker."""

    model_config = ConfigDict(extra="ignore")
    dictionary: str = Field("DICT_4X4_50", description="ArUco dictionary name")
    aruco_id: Optional[int] = Field(
        None, description="ArUco marker id; defaults to the next available id if omitted"
    )
    size: float = Field(0.021, description="Marker side length in meters")
    border_width: float = Field(0.05, description="Marker border width fraction")
    position: Vec3 = Field(default_factory=Vec3, description="Marker position in world frame")
    normal: NormalVec3 = Field(
        default_factory=NormalVec3, description="Surface normal in world frame"
    )


class CadPose(BaseModel):
    """Body for POST /api/cad-pose."""

    model_config = ConfigDict(extra="ignore")
    position: Vec3 = Field(default_factory=Vec3, description="CAD object position")
    rotation: RPY = Field(default_factory=RPY, description="CAD object orientation (deg)")


class RotationUpdate(BaseModel):
    """Body for PATCH /api/markers/{marker_id}/rotation."""

    model_config = ConfigDict(extra="ignore")
    mode: str = Field("relative", description="'relative' (delta) or 'absolute'")
    yaw: float = Field(0.0, description="In-plane yaw in degrees")


class TranslationUpdate(BaseModel):
    """Body for PATCH /api/markers/{marker_id}/translation."""

    model_config = ConfigDict(extra="ignore")
    mode: str = Field("relative", description="'relative' (delta) or 'absolute'")
    x: float = Field(0.0, description="In-plane X translation")
    y: float = Field(0.0, description="In-plane Y translation")


class SwapRequest(BaseModel):
    """Body for POST /api/markers/swap."""

    model_config = ConfigDict(extra="ignore")
    marker1_id: Optional[int] = Field(None, description="First marker's internal id")
    marker2_id: Optional[int] = Field(None, description="Second marker's internal id")
