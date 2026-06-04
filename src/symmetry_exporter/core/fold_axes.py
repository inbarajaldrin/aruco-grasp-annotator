"""
Convert a detected FoldSymmetry into the symmetry_exporter ``fold_axes`` JSON schema.

The schema (matching ``data/symmetry/{object}_symmetry.json``) lists all three mesh-local
axes; for a fold-N axis the ``quaternions`` are the N rotations k·(360/N)° about that axis.
This reproduces the previously human-annotated files byte-for-byte (verified on all 12
existing objects), so auto-detection is a drop-in for the manual annotation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .symmetry_detector import detect_symmetry_robust

_AXES = ("x", "y", "z")
_AXIS_UNIT = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

# detect_symmetry_robust marks a continuous (revolution) axis with folds[ax] == its
# continuous_sweep sentinel (72). The discrete fold_axes schema can't represent that.
_CONTINUOUS_SENTINEL = 72


def _clean(v: float) -> float | int:
    """Round to 6 dp and collapse exact integers to int (matches the existing files)."""
    r = round(v, 6)
    return int(r) if r == int(r) else r


def quaternion_about_axis(axis: str, angle_deg: float) -> dict[str, Any]:
    """Quaternion {x,y,z,w} for a rotation of ``angle_deg`` about a principal axis."""
    ux, uy, uz = _AXIS_UNIT[axis]
    half = math.radians(angle_deg) / 2.0
    s = math.sin(half)
    return {"x": _clean(ux * s), "y": _clean(uy * s), "z": _clean(uz * s), "w": _clean(math.cos(half))}


def folds_to_fold_axes(folds: dict[str, int], object_name: str) -> dict[str, Any]:
    """Build the full fold_axes record from a {axis: fold} map (folds >= 2 only)."""
    fold_axes: dict[str, Any] = {}
    for ax in _AXES:
        n = int(folds.get(ax, 1))
        if n >= _CONTINUOUS_SENTINEL:
            raise ValueError(
                f"{object_name}: axis {ax} is continuous (revolution) symmetric; "
                f"the fold_axes schema only represents discrete cyclic folds"
            )
        quaternions = [
            {
                "angle_deg": int(round(k * 360.0 / n)),
                "quaternion": quaternion_about_axis(ax, k * 360.0 / n),
            }
            for k in range(n)
        ]
        fold_axes[ax] = {"fold": n, "quaternions": quaternions}
    return {"object_name": object_name, "fold_axes": fold_axes}


def auto_detect_fold_axes(mesh_path: str | Path, object_name: str) -> dict[str, Any]:
    """Run the deterministic detector on a CAD mesh and return the fold_axes record."""
    sym = detect_symmetry_robust(str(mesh_path), obj_id=0)
    folds = {ax: f for ax, f in sym.folds.items() if f >= 2}
    return folds_to_fold_axes(folds, object_name)
