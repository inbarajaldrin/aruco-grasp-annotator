"""
Typed request body for Symmetry Exporter endpoints.

Mirrors the prior loose ``dict`` body. ``object_name`` stays Optional so the handler's
own "required" 400 check is preserved. ``fold_axes`` is kept broad (an arbitrary nested
mapping) because the handler persists it verbatim — over-modeling it would drop fields.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExportSymmetryRequest(BaseModel):
    """Body for POST /api/export-symmetry."""

    model_config = ConfigDict(extra="ignore")
    object_name: Optional[str] = Field(None, description="Object/model name")
    fold_axes: Dict[str, Any] = Field(
        default_factory=dict, description="Fold-symmetry axes (arbitrary nested mapping)"
    )
