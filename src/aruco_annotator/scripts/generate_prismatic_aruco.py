#!/usr/bin/env python3
"""
Generate deterministic ArUco annotations for prismatic objects into data_v2/aruco/.

Covers box and U-family objects: one marker per primary face, placed at the centre and
slid toward a corner only as far as needed to fully seat on material (the hand process,
codified in core.aruco_placement). IDs flow annotation -> real world (markers numbered
id_base.., physical markers printed to match). Existing data/ is never touched.

Flat boards (base*) and small-facet hexes are out of scope here (handled separately).

Usage (from repo root):
    .venv/bin/python src/aruco_annotator/scripts/generate_prismatic_aruco.py [obj ...] [--id-base N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `aruco_annotator` importable

from aruco_annotator.core.aruco_placement import generate_prismatic_aruco  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_MODELS = _REPO / "data" / "models"
_OUT = _REPO / "data_v2" / "aruco"
_MODEL_EXTS = (".obj", ".stl", ".ply")

# Box + U-family prismatic objects (place-then-slide-to-seat reproduces them at full contact).
PRISMATIC_OBJECTS = [
    "fork_orange", "fork_yellow",
    "line_brown", "line_green", "line_red",
    "u_brown", "u_green", "u_orange",
    "inverted_u_brown", "inverted_u_yellow",
]


def _find_model(name: str) -> Path | None:
    for ext in _MODEL_EXTS:
        candidate = _MODELS / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("objects", nargs="*", help="object names (default: the prismatic box+U set)")
    ap.add_argument("--id-base", type=int, default=0, help="first aruco id per object")
    args = ap.parse_args()

    objects = args.objects or PRISMATIC_OBJECTS
    _OUT.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for name in objects:
        mesh = _find_model(name)
        if mesh is None:
            print(f"  {name:18} SKIP (no model in data/models)")
            n_skip += 1
            continue
        record = generate_prismatic_aruco(mesh, name, id_base=args.id_base)
        (_OUT / f"{name}_aruco.json").write_text(json.dumps(record, indent=2))
        faces = [m["face_type"] for m in record["markers"]]
        dropped = [s["face_type"] for s in record.get("skipped_faces", [])]
        note = f"  (skipped {dropped})" if dropped else ""
        print(f"  {name:18} {len(record['markers'])} markers size={record['size'] * 1000:.0f}mm {faces}{note}")
        n_ok += 1

    print(f"\n{n_ok} generated, {n_skip} skipped -> {_OUT}")
    return 1 if n_skip else 0


if __name__ == "__main__":
    raise SystemExit(main())
