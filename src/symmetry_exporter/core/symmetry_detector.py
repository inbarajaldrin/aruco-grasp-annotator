# Reference: pose6d/core/fusion/cad_symmetry.py (dual-a4500:~/Documents/pose6d), vendored 2026-06-04.
# Deterministic, geometry-only CAD fold-symmetry detector. Vendored VERBATIM and kept STRICT per
# pose6d ADR-0006: relaxing the thresholds produces false-positive symmetry that silently corrupts
# every downstream pose. This app uses detect_symmetry_robust + FoldSymmetry; the canonicalization
# helpers come along unchanged and unused.

# Lifted from aic_vision/aic_vision/fusion/cad_symmetry.py (re-vendor-lazy, slice 3a).
"""CAD fold-symmetry classification + symmetry-aware pose canonicalization.

Background
==========
Some tracked CADs are fold-symmetric about one or more link-frame axes
(e.g. the SC port is fold-2 symmetric about its Y_link / port-axis). FP
register/seeded_track running on a symmetric CAD has a true ambiguity:
the rotation R_cam_obj and R_cam_obj @ R_localsym are visually
indistinguishable, so the local optimizer snaps to whichever basin is
closer at init. Live observation 2026-05-19 on SC: ~2/3 of FP outputs
land in the 180°-flipped basin, breaking yaw clustering.

This module supplies two pieces:

  1. ``detect_fold_symmetry(mesh_path)`` — at CAD-onboarding time,
     auto-detect fold values about each link-frame axis by sampling the
     surface and Chamfer-comparing the cloud against its rotated copy.
     Writes a ``symmetry.json`` alongside the mesh.

  2. ``canonicalize_to_anchor(R_cam_obj, R_cam_obj_anchor, sym)`` — at
     runtime, given an FP candidate rotation and a trusted anchor
     (typically the seed pose), return the symmetry-equivalent
     representative of ``R_cam_obj`` that is closest to the anchor.
     This makes the 180°-flipped candidate snap to its in-basin twin
     before clustering / fusion.

Reference for the fold-symmetry concept: the
``aruco_camera_localizer`` (in operator's ~/Documents) uses
per-object ``fold_axes`` JSON files and a ``snap_orientation_to_cardinal``
step. We adopt the same data shape so the registry idea is reusable.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


_R_BY_AXIS = {"x": _Rx, "y": _Ry, "z": _Rz}


def _rot_geodesic_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_t = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_t)))


# ---------------------------------------------------------------------------
# Symmetry data class
# ---------------------------------------------------------------------------
@dataclass
class FoldSymmetry:
    """Per-object fold-symmetry record.

    ``folds`` maps the OBJECT-LOCAL link-frame axis name ("x"/"y"/"z")
    to the integer fold value (1 = no symmetry; 2 = 180° symmetric; etc.).
    Only axes with fold >= 2 are recorded.

    ``chamfer_mm`` carries the measured Chamfer distance for each fold
    (smaller = stronger symmetry; useful for diagnostics).
    """

    obj_id: int
    folds: dict[str, int] = field(default_factory=dict)
    chamfer_mm: dict[str, float] = field(default_factory=dict)
    bbox_extent_mm: float = 0.0
    baseline_chamfer_mm: float = 0.0
    source: str = ""
    # Object-local axis name ("x"/"y"/"z") of a CONTINUOUS (revolution) symmetry,
    # if any (set by detect_symmetry_robust). None = purely discrete / asymmetric.
    continuous_axis: Optional[str] = None

    @property
    def is_symmetric(self) -> bool:
        return any(f >= 2 for f in self.folds.values())

    def strong_folds(
        self, max_chamfer_ratio: float = 3.0
    ) -> dict[str, int]:
        """Subset of folds whose measured chamfer is strong evidence.

        ``chamfer_mm[axis] / baseline_chamfer_mm <= max_chamfer_ratio``
        keeps only symmetries where the rotated cloud sits well within
        intra-sample sampling noise. Borderline detections (NIC's
        Z-fold-2 at ~4.6× baseline) are filtered out — applying
        canonicalization for those would collapse the basin distinction
        the daemon uses for rejection.
        """
        if self.baseline_chamfer_mm <= 0:
            return {}
        return {
            axis: fold
            for axis, fold in self.folds.items()
            if self.chamfer_mm.get(axis, float("inf"))
            <= max_chamfer_ratio * self.baseline_chamfer_mm
        }

    def with_strong_folds_only(
        self, max_chamfer_ratio: float = 3.0
    ) -> "FoldSymmetry":
        """Return a copy that only retains strong folds. Useful at the
        daemon load-site to gate which symmetries are applied."""
        strong = self.strong_folds(max_chamfer_ratio)
        return FoldSymmetry(
            obj_id=self.obj_id,
            folds=dict(strong),
            chamfer_mm={k: v for k, v in self.chamfer_mm.items() if k in strong},
            bbox_extent_mm=self.bbox_extent_mm,
            baseline_chamfer_mm=self.baseline_chamfer_mm,
            source=self.source,
            continuous_axis=self.continuous_axis if self.continuous_axis in strong else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obj_id": int(self.obj_id),
            "folds": dict(self.folds),
            "chamfer_mm": {k: float(v) for k, v in self.chamfer_mm.items()},
            "bbox_extent_mm": float(self.bbox_extent_mm),
            "baseline_chamfer_mm": float(self.baseline_chamfer_mm),
            "source": str(self.source),
            "continuous_axis": self.continuous_axis,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FoldSymmetry":
        return cls(
            obj_id=int(d["obj_id"]),
            folds={k: int(v) for k, v in d.get("folds", {}).items()},
            chamfer_mm={k: float(v) for k, v in d.get("chamfer_mm", {}).items()},
            bbox_extent_mm=float(d.get("bbox_extent_mm", 0.0)),
            baseline_chamfer_mm=float(d.get("baseline_chamfer_mm", 0.0)),
            source=str(d.get("source", "")),
            continuous_axis=d.get("continuous_axis"),
        )


# ---------------------------------------------------------------------------
# Detection: Chamfer-based fold-symmetry test
# ---------------------------------------------------------------------------
def _chamfer_one_way(A: np.ndarray, B: np.ndarray, sample: int, rng: np.random.Generator) -> float:
    a_idx = rng.choice(len(A), size=min(sample, len(A)), replace=False)
    b_idx = rng.choice(len(B), size=min(sample, len(B)), replace=False)
    A_s = A[a_idx]
    B_s = B[b_idx]
    d_mins: list[np.ndarray] = []
    for chunk in np.array_split(A_s, 8):
        d2 = ((chunk[:, None, :] - B_s[None, :, :]) ** 2).sum(-1)
        d_mins.append(np.sqrt(d2.min(axis=1)))
    return float(np.mean(np.concatenate(d_mins)))


def _chamfer_sym(A: np.ndarray, B: np.ndarray, sample: int, rng: np.random.Generator) -> float:
    return 0.5 * (_chamfer_one_way(A, B, sample, rng) + _chamfer_one_way(B, A, sample, rng))


def detect_fold_symmetry(
    mesh_path: str | Path,
    obj_id: int,
    n_pts: int = 8000,
    sample: int = 4000,
    folds_to_test: tuple[int, ...] = (2, 3, 4, 6),
    chamfer_safety_factor: float = 5.0,
    seed: int = 0,
) -> FoldSymmetry:
    """Auto-detect fold symmetries of a CAD mesh about its link-frame axes.

    The mesh is sampled, the cloud is rotated about each (X, Y, Z) axis
    through the cloud centroid by each candidate fold angle, and the
    Chamfer distance to the original is computed. An axis is declared
    fold-N symmetric if the rotated-vs-original chamfer is within
    ``chamfer_safety_factor × baseline``, where baseline is the chamfer
    between two disjoint sub-samples of the original cloud.

    Sub-symmetries are pruned: e.g. fold-6 implies fold-2 and fold-3.
    Only the highest standalone fold per axis is kept.
    """
    import trimesh  # local import to keep daemon-side import light

    mesh = trimesh.load(str(mesh_path), force="mesh")
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_pts)
    pts = np.asarray(pts, dtype=np.float64)
    rng = np.random.default_rng(seed)

    bbox_extent = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    sub1 = rng.choice(len(pts), min(sample, len(pts)), replace=False)
    sub2 = rng.choice(len(pts), min(sample, len(pts)), replace=False)
    baseline = _chamfer_sym(pts[sub1], pts[sub2], sample, rng)
    accept_threshold = chamfer_safety_factor * baseline

    centroid = pts.mean(axis=0)
    folds: dict[str, int] = {}
    chamfer_mm: dict[str, float] = {}
    raw: dict[str, dict[int, float]] = {}

    for axis_name, R_fn in _R_BY_AXIS.items():
        raw[axis_name] = {}
        passing: list[int] = []
        for fold in folds_to_test:
            angle = 2 * math.pi / fold
            R = R_fn(angle)
            rotated = (pts - centroid) @ R.T + centroid
            d = _chamfer_sym(pts, rotated, sample, rng)
            raw[axis_name][fold] = d
            if d <= accept_threshold:
                passing.append(fold)

        # Sub-symmetry prune: keep the largest standalone fold. fold-6
        # being passing alongside fold-2 and fold-3 implies a true fold-6
        # symmetry; but if ONLY fold-2 + fold-6 pass (no fold-3), the
        # fold-6 is a coincidence — keep fold-2. Algorithm: iterate
        # largest→smallest, keep the first fold whose proper divisors
        # ALSO pass.
        chosen_fold = 1
        for cand in sorted(passing, reverse=True):
            divisors_ok = all(
                d_ in passing for d_ in (2, 3) if d_ < cand and cand % d_ == 0
            )
            if divisors_ok:
                chosen_fold = cand
                break
        if chosen_fold >= 2:
            folds[axis_name] = chosen_fold
            chamfer_mm[axis_name] = raw[axis_name][chosen_fold] * 1000.0

    sym = FoldSymmetry(
        obj_id=obj_id,
        folds=folds,
        chamfer_mm=chamfer_mm,
        bbox_extent_mm=bbox_extent * 1000.0,
        baseline_chamfer_mm=baseline * 1000.0,
        source=str(mesh_path),
    )
    logger.info(
        "detect_fold_symmetry obj_id=%d folds=%s chamfer_mm=%s baseline=%.3fmm",
        obj_id, folds, {k: round(v, 3) for k, v in chamfer_mm.items()}, baseline * 1000,
    )
    return sym


# ---------------------------------------------------------------------------
# Per-CAD persistence
# ---------------------------------------------------------------------------
def write_symmetry_json(sym: FoldSymmetry, mesh_path: str | Path) -> Path:
    """Write ``symmetry.json`` next to the mesh file. Returns the path."""
    out = Path(mesh_path).parent / "symmetry.json"
    out.write_text(json.dumps(sym.to_dict(), indent=2))
    return out


def load_symmetry_json(mesh_path: str | Path) -> Optional[FoldSymmetry]:
    """Load the ``symmetry.json`` next to a mesh file, if present."""
    p = Path(mesh_path).parent / "symmetry.json"
    if not p.is_file():
        return None
    try:
        return FoldSymmetry.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("failed to load symmetry json %s: %s", p, e)
        return None


# ---------------------------------------------------------------------------
# Canonicalization: snap a candidate rotation to its symmetry-equivalent
# representative closest to a trusted anchor.
# ---------------------------------------------------------------------------
def equivalent_rotations(R: np.ndarray, sym: FoldSymmetry) -> list[np.ndarray]:
    """Enumerate all rotations equivalent to ``R`` under the fold-symmetry
    group of the object. Always includes ``R`` itself.

    The fold-symmetry transforms are applied in the **object-local
    frame**: each rotation is ``R @ G`` (right-multiplication).

    The local symmetry transforms ``G`` are the full finite proper-rotation
    GROUP CLOSURE generated by the per-axis fold rotations — not each axis
    enumerated independently. For a single dominant fold axis (the AIC case)
    the closure equals that one cyclic group (unchanged); for multi-axis
    symmetry (e.g. a cube ``{x:4,y:4,z:4}`` → order-24 group) the composed
    elements are included, so a pose equivalent only through a *composed*
    rotation canonicalizes correctly (codex review fix).
    """
    gens = [
        _R_BY_AXIS[ax](2 * math.pi / fold)
        for ax, fold in sym.folds.items() if fold >= 2
    ]
    group = [np.eye(3)]
    frontier = list(gens)
    while frontier:
        G = frontier.pop()
        if any(np.linalg.norm(G - H) < 1e-6 for H in group):
            continue
        group.append(G)
        if len(group) > 200:  # safety cap (a real finite rotation group is ≤ 60)
            break
        for g in gens:
            frontier.append(G @ g)
    return [R @ G for G in group]


def canonicalize_to_anchor(
    R: np.ndarray,
    R_anchor: np.ndarray,
    sym: Optional[FoldSymmetry],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the symmetry-equivalent of ``R`` closest to ``R_anchor``.

    If ``sym`` is None or asymmetric, ``R`` is returned unchanged. The
    returned dict carries diagnostics (which equivalent was picked,
    geodesic distance before vs after, total alternatives considered).
    """
    if sym is None or not sym.is_symmetric:
        return R.copy(), {
            "snapped": False,
            "n_equivalents": 1,
            "geodesic_before_deg": math.degrees(_rot_geodesic_rad(R, R_anchor)),
            "geodesic_after_deg": math.degrees(_rot_geodesic_rad(R, R_anchor)),
            "winner_idx": 0,
        }

    equivs = equivalent_rotations(R, sym)
    dists = [_rot_geodesic_rad(e, R_anchor) for e in equivs]
    winner = int(np.argmin(dists))
    R_canon = equivs[winner]
    return R_canon, {
        "snapped": winner != 0,
        "n_equivalents": len(equivs),
        "geodesic_before_deg": math.degrees(dists[0]),
        "geodesic_after_deg": math.degrees(dists[winner]),
        "winner_idx": winner,
        "all_geodesics_deg": [math.degrees(d) for d in dists],
    }


# ---------------------------------------------------------------------------
# Convenience: 4x4 transform variant
# ---------------------------------------------------------------------------
def canonicalize_T_to_anchor(
    T: np.ndarray,
    T_anchor: np.ndarray,
    sym: Optional[FoldSymmetry],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Same as ``canonicalize_to_anchor`` but operates on full 4x4
    transforms. The translation is left unchanged (fold symmetry about
    an axis through the link origin doesn't move the origin)."""
    R_canon, diag = canonicalize_to_anchor(T[:3, :3], T_anchor[:3, :3], sym)
    T_out = T.copy()
    T_out[:3, :3] = R_canon
    return T_out, diag


# ───────────────────────────────────────────────────────────────────────────
# Robust geometry-only symmetry auto-ingestion (CAD → FoldSymmetry, no human).
#
# Replaces the mean-chamfer detector's weaknesses (it washes out small
# symmetry-breaking features and confuses 4-vs-6) with the GPT-designed recipe:
#   • metric = high-PERCENTILE (p99.5) + MAX nearest-neighbour distance + surface
#     NORMAL agreement (so a tiny deliberate breaker dominates the decision),
#   • SCALE-AWARE thresholds (tied to object diameter AND a sampling-noise baseline),
#   • EXACT cyclic-group validation (fold N accepted only if EVERY k·360/N rotation
#     passes → kills the 4-vs-6 confusion), maximal valid fold per axis,
#   • proper rotations only (det +1; mirror symmetry is irrelevant to a rigid pose),
#   • CROSS-AXIS group-closure validation (every composed element must also pass),
#   • CONTINUOUS (revolution) detection via a fine angular sweep.
# Detection is in the mesh-local x/y/z frame (the convention FoldSymmetry/BOP use);
# arbitrary-frame CAD (symmetry axis not aligned to local x/y/z) is a noted extension.
# ───────────────────────────────────────────────────────────────────────────
def _axis_unit(name: str) -> np.ndarray:
    return {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}[name]


def _rot_about(axis_unit: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues proper rotation (det = +1) of angle theta about a unit axis."""
    a = axis_unit / (np.linalg.norm(axis_unit) + 1e-12)
    x, y, z = a
    c, s, C = math.cos(theta), math.sin(theta), 1 - math.cos(theta)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _sample_surface(mesh, n_pts: int, seed: int):
    """Dense sampled surface points (DETERMINISTIC via the seeded RNG). Surface-only,
    no normals: the detector is position-only (see detect_symmetry_robust) — mesh face
    normals are unreliable on non-watertight / degenerate real-world CAD (they broke
    detection on aic's SC port), and a tight position-residual threshold + exact group
    validation separates true from spurious symmetry without them."""
    import trimesh
    # Seed the global RNG around the sample so the same (mesh, seed) is reproducible —
    # trimesh.sample.sample_surface draws from np.random; restore state afterwards.
    st = np.random.get_state()
    try:
        np.random.seed(seed)
        pts, _ = trimesh.sample.sample_surface(mesh, n_pts)
    finally:
        np.random.set_state(st)
    return np.asarray(pts, dtype=np.float64)


def _residual_R(P, tree, center, R):
    """POSITION-only residual: p99.5 of the nearest-neighbour DISTANCE from the cloud
    transformed by proper rotation R (about center) to the original (tree on original P).
    A true symmetry sits at ~1× the sampling baseline; a spurious one (or a small spatial
    symmetry-breaker, whose points go unmatched) sits well above it. Position-only by
    design — mesh-normal cues are unreliable on real-world non-watertight CAD (they broke
    aic's SC port); a tight baseline-relative threshold + exact group validation separate
    true from spurious without them (true ≤ ~1.2× baseline, spurious ≥ ~1.7× across the
    FMB + aic objects)."""
    P_rot = (P - center) @ R.T + center
    d, _ = tree.query(P_rot, k=1)
    return float(np.percentile(d, 99.5))


def _residual(P, tree, center, axis_u, theta):
    """Position residual for a rotation of angle theta about unit axis axis_u."""
    return _residual_R(P, tree, center, _rot_about(axis_u, theta))


def detect_symmetry_robust(
    mesh_path: str | Path,
    obj_id: int = 0,
    n_pts: int = 12000,
    folds_to_test: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12),
    pos_diam_frac: float = 0.001,
    pos_baseline_factor: float = 1.45,
    continuous_sweep: int = 72,
    seed: int = 0,
) -> FoldSymmetry:
    """Auto-detect an object's proper-rotation fold symmetry from CAD geometry alone.

    Deterministic (seeded sampling). POSITION-only metric (p99.5 NN distance) with a tight
    baseline-relative threshold ``tau_pos = max(pos_diam_frac*diameter,
    pos_baseline_factor*baseline_p995)`` + exact cyclic-group validation + cross-axis
    group-closure. Returns a :class:`FoldSymmetry` (``source='robust-geometry'``): ``folds``
    = the maximal valid cyclic fold per mesh-local axis; ``continuous_axis`` if a revolution
    symmetry is found; ``chamfer_mm`` / ``baseline_chamfer_mm`` populated so the strong-fold
    gate is meaningful. No texture/colour — symmetry is geometry.

    Resolvable-feature limit (honest): a symmetry-breaking feature smaller than ~0.5% of the
    sampled surface falls below the p99.5 tail and may be missed — raise ``n_pts`` / the
    percentile to resolve finer features. (FoundationPose likewise cannot resolve
    sub-sensor-resolution features, so this matches the estimator's own limit.)
    """
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(str(mesh_path), force="mesh")
    diameter = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    P = _sample_surface(mesh, n_pts, seed)
    tree = cKDTree(P)

    # Sampling-noise baseline: an independent sample of the SAME mesh, no rotation.
    d_bl, _ = tree.query(_sample_surface(mesh, n_pts, seed + 1), k=1)
    base_p995 = float(np.percentile(d_bl, 99.5))

    tau_pos = max(pos_diam_frac * diameter, pos_baseline_factor * base_p995)

    # Center: centroid is reliable even for non-watertight CAD; bbox center as an alternative.
    centroid = P.mean(axis=0)
    bbox_c = 0.5 * (mesh.bounds[0] + mesh.bounds[1])

    def passes(center, axis_u, theta) -> bool:
        if abs((theta % (2 * math.pi))) < 1e-6:
            return True
        return _residual(P, tree, center, axis_u, theta) <= tau_pos

    folds: dict[str, int] = {}
    chamfer_mm: dict[str, float] = {}
    continuous_axis: Optional[str] = None

    for ax in ("x", "y", "z"):
        axis_u = _axis_unit(ax)
        # Pick the better center for THIS axis by the 180° residual.
        center = min((centroid, bbox_c), key=lambda c: _residual(P, tree, c, axis_u, math.pi))

        # Continuous check first (fine sweep all-pass ⇒ revolution about this axis).
        sweep = [2 * math.pi * k / continuous_sweep for k in range(1, continuous_sweep)]
        if all(passes(center, axis_u, t) for t in sweep):
            continuous_axis = ax
            folds[ax] = continuous_sweep  # sentinel: very-high fold ⇒ treat as continuous
            chamfer_mm[ax] = _residual(P, tree, center, axis_u, math.pi) * 1000.0
            continue

        # Exact cyclic-group validation: fold N valid iff EVERY k·2π/N rotation passes.
        valid = [
            n for n in folds_to_test
            if all(passes(center, axis_u, 2 * math.pi * k / n) for k in range(1, n))
        ]
        if valid:
            fold = max(valid)
            folds[ax] = fold
            chamfer_mm[ax] = _residual(P, tree, center, axis_u, 2 * math.pi / fold) * 1000.0

    # Cross-axis group-closure validation: build the proper-rotation group generated by
    # the accepted per-axis folds; require EVERY composed element to also pass the metric.
    # If the closure fails (an impossible combo slipped through per-axis), greedily downgrade.
    def closure_ok(sel: dict[str, int]) -> bool:
        gens = [_rot_about(_axis_unit(a), 2 * math.pi / f) for a, f in sel.items() if f < continuous_sweep]
        if not gens:
            return True
        group = [np.eye(3)]
        frontier = list(gens)
        while frontier:
            R = frontier.pop()
            if any(np.linalg.norm(R - G) < 1e-6 for G in group):
                continue
            group.append(R)
            if len(group) > 120:
                return False  # blew up ⇒ not a valid finite symmetry combo
            for g in gens:
                frontier.append(R @ g)
        # every non-identity element (a rotation matrix) must itself be a real symmetry.
        for R in group:
            if np.linalg.norm(R - np.eye(3)) < 1e-6:
                continue
            if _residual_R(P, tree, centroid, R) > tau_pos:
                return False
        return True

    discrete = {a: f for a, f in folds.items() if f < continuous_sweep}
    if discrete and not closure_ok(discrete):
        # Downgrade: keep the strongest single axis, then add axes one at a time while closure holds.
        ordered = sorted(discrete.items(), key=lambda kv: -kv[1])
        kept: dict[str, int] = {ordered[0][0]: ordered[0][1]}
        for a, f in ordered[1:]:
            trial = dict(kept); trial[a] = f
            if closure_ok(trial):
                kept = trial
        folds = {**{a: f for a, f in folds.items() if f >= continuous_sweep}, **kept}
        chamfer_mm = {a: chamfer_mm[a] for a in folds if a in chamfer_mm}

    return FoldSymmetry(
        obj_id=obj_id,
        folds=folds,
        chamfer_mm=chamfer_mm,
        bbox_extent_mm=diameter * 1000.0,
        baseline_chamfer_mm=base_p995 * 1000.0,
        source="robust-geometry",
        continuous_axis=continuous_axis,
    )
