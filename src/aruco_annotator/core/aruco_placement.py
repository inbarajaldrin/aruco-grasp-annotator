"""
Deterministic ArUco placement for prismatic objects (box + U + …), geometry-driven.

The unified rule (the author's hand process, codified):
  1. Consider the 6 primary faces. For each, lay a marker at the face centre.
  2. SEATING TEST: ray-cast the marker's square footprint inward along the face
     normal; the marker seats iff the whole footprint is backed by material at the
     face plane. A box's centre seats; a U's centre dangles over the notch.
  3. If the centre doesn't seat, slide the marker to the first CORNER (inset by half
     the marker, canonical order) that fully seats — landing it on solid material.
  4. If nothing seats (e.g. a hex's tiny side face), skip that face.

Marker +Z aligns with the outward face normal (deterministic from the normal).
IDs flow annotation -> real world: markers are numbered id_base.. and the physical
markers are printed to match. Output is the data/aruco schema.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Printable ArUco side lengths (m), largest first. 21 mm default; smaller only when it won't fit.
PRINTABLE_SIZES_M = (0.021, 0.014, 0.010, 0.007)

# Canonical face order (matches place-all-6) → (normal, normal axis, sign, in-plane u axis, v axis)
_FACES = [
    ([-1.0, 0.0, 0.0], 0, -1, 1, 2),
    ([1.0, 0.0, 0.0], 0, 1, 1, 2),
    ([0.0, -1.0, 0.0], 1, -1, 0, 2),
    ([0.0, 1.0, 0.0], 1, 1, 0, 2),
    ([0.0, 0.0, -1.0], 2, -1, 0, 1),
    ([0.0, 0.0, 1.0], 2, 1, 0, 1),
]


def fit_marker_size(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    """Largest printable marker that fits the smallest bbox extent (21 mm default)."""
    min_extent = float((bbox_max - bbox_min).min())
    for size in PRINTABLE_SIZES_M:
        if size <= min_extent + 1e-9:
            return size
    return PRINTABLE_SIZES_M[-1]


def _build_raycasting_scene(o3d_mesh):
    """Build an open3d RaycastingScene from a legacy TriangleMesh (no extra deps)."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))
    return scene


# Fraction of the marker footprint that must rest on material for the marker to "seat".
# Not 1.0: real faces have gaps (a fork's prongs leave ~80% support at its best spot),
# while a marker over a void (a U-notch centre) sits at ~0-11%. 0.75 separates them.
DEFAULT_MIN_SUPPORT = 0.60


def support_fraction(
    scene,
    face_center: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    size: float,
    *,
    grid: int = 9,
    tol: float | None = None,
) -> float:
    """Fraction of the size×size marker footprint backed by material at the face plane.

    Midpoint-rule area estimate: the footprint is split into ``grid×grid`` cells and one
    ray is cast through each cell's CENTRE, inward along −normal from one marker-width
    outside the face. A cell counts as supported iff its ray hits the outer surface after
    travelling ``size`` (within ``tol``). Cell-centre sampling means a marker sitting flush
    with a face edge scores 1.0 (every cell centre is inside the material — the outermost
    is inset by half a cell, size/(2·grid), by construction), while a real void (a U-notch
    or a narrow leg) is measured as its true uncovered area fraction.
    """
    import open3d as o3d

    if tol is None:
        tol = max(0.0005, 0.02 * size)  # 0.5 mm or 2% of the marker
    half = size / 2.0
    cell = size / grid
    ts = -half + (np.arange(grid) + 0.5) * cell  # cell centres (midpoint rule)
    rays = np.array(
        [
            [*(face_center + du * u_axis + dv * v_axis + normal * size), *(-normal)]
            for du in ts
            for dv in ts
        ],
        dtype=np.float32,
    )
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return float(np.mean(np.abs(t_hit - size) <= tol))


def marker_seats(
    scene,
    face_center: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    size: float,
    *,
    min_support: float = DEFAULT_MIN_SUPPORT,
    grid: int = 9,
) -> bool:
    """True iff at least ``min_support`` of the marker footprint rests on material."""
    return support_fraction(scene, face_center, normal, u_axis, v_axis, size, grid=grid) >= min_support


# Canonical corner directions to slide toward (sign on u, sign on v).
_SLIDE_DIRS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def _find_seat(
    scene,
    face_center: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    half_u: float,
    half_v: float,
    half_mk: float,
    size: float,
    *,
    min_support: float,
    full_bar: float = 0.999,
    steps: int = 24,
):
    """Find where to seat the marker on a face, modelling the hand process.

    Keep the centre if it is fully seated; otherwise slide out toward each corner
    (canonical order) and stop at the FIRST position that is fully seated — the marker
    just clears a void without overshooting onto a narrow tip. If nothing reaches full
    seating, fall back to the best-supported spot (>= ``min_support``), else give up.

    Returns the chosen position (np.ndarray) or None if even the best spot dangles.
    """
    if support_fraction(scene, face_center, normal, u_axis, v_axis, size) >= full_bar:
        return face_center

    ou = max(0.0, half_u - half_mk)  # in-plane slide room toward a corner
    ov = max(0.0, half_v - half_mk)

    best_pos, best_sup = None, -1.0
    for su, sv in _SLIDE_DIRS:
        for k in range(1, steps + 1):
            t = k / steps
            pos = face_center + (su * ou * t) * u_axis + (sv * ov * t) * v_axis
            sup = support_fraction(scene, pos, normal, u_axis, v_axis, size)
            if sup >= full_bar:
                return pos  # first fully-seated position along this slide
            if sup > best_sup:
                best_sup, best_pos = sup, pos

    return best_pos if best_sup >= min_support else None


def generate_prismatic_aruco(
    mesh_path: str | Path,
    object_name: str,
    *,
    size: float | None = None,
    border_width: float = 0.05,
    dictionary: str = "DICT_4X4_50",
    id_base: int = 0,
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> dict[str, Any]:
    """Seating-aware ArUco placement: a marker per face, slid to a seated corner if needed."""
    from ..core.cad_loader import CADLoader
    from ..models.marker import MarkerData
    from ..services.mesh_service import determine_face_type

    loader = CADLoader()
    mesh = loader.load_file(Path(mesh_path), input_units="auto")
    mesh_info = loader.get_mesh_info(mesh)
    scene = _build_raycasting_scene(mesh)

    bbox_min = np.array(mesh_info["bbox_min"])
    bbox_max = np.array(mesh_info["bbox_max"])
    center = (bbox_min + bbox_max) / 2.0
    if size is None:
        size = fit_marker_size(bbox_min, bbox_max)
    half_mk = size / 2.0

    markers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for normal_list, n_idx, sign, u_idx, v_idx in _FACES:
        normal = np.array(normal_list)
        u_axis = np.eye(3)[u_idx]
        v_axis = np.eye(3)[v_idx]
        face_type = determine_face_type(tuple(normal))
        half_u = float(bbox_max[u_idx] - bbox_min[u_idx]) / 2.0
        half_v = float(bbox_max[v_idx] - bbox_min[v_idx]) / 2.0

        if half_u < half_mk - 1e-9 or half_v < half_mk - 1e-9:
            skipped.append({"face_type": face_type, "reason": "marker larger than face"})
            continue

        face_center = center.copy()
        face_center[n_idx] = bbox_max[n_idx] if sign > 0 else bbox_min[n_idx]

        placed = _find_seat(
            scene, face_center, normal, u_axis, v_axis,
            half_u, half_v, half_mk, size, min_support=min_support,
        )
        if placed is None:
            skipped.append({"face_type": face_type, "reason": "no seated position found"})
            continue

        marker = MarkerData(
            aruco_id=id_base + len(markers),
            dictionary=dictionary,
            size=size,
            border_width=border_width,
            position=tuple(placed),
            face_normal=tuple(normal),
            face_type=face_type,
        )
        markers.append(
            {
                "aruco_id": marker.aruco_id,
                "face_type": marker.face_type,
                "surface_normal": marker.face_normal.tolist(),
                "T_object_to_marker": marker.get_T_object_to_marker(center.tolist()),
            }
        )

    record = {
        "exported_at": datetime.now().isoformat(),
        "model_file": f"{object_name}{Path(mesh_path).suffix}",
        "total_markers": len(markers),
        "aruco_dictionary": dictionary,
        "size": size,
        "border_width": border_width,
        "cad_object_info": {
            "center": center.tolist(),
            "dimensions": mesh_info["dimensions"],
            "position": [0.0, 0.0, 0.0],
            "rotation": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        "markers": markers,
        "notes": "T_object_to_marker is the complete transform from object center to marker.",
    }
    if skipped:
        record["skipped_faces"] = skipped
    return record
