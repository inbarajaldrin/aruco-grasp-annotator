#!/usr/bin/env python3
"""
Config-driven deterministic ArUco generation into data_v2/aruco/.

Reads the assembly configs (data/fmb_assembly*.json) and, for each component, applies the
placement rule selected by its existing type/subtype — no per-object hand-flagging:

    type == "board"        -> top (+z) face, 4 corner markers
    subtype == "peg" (hex) -> SKIPPED for now (hex rule not implemented yet)
    block / socket / other -> prismatic slide-seat (marker per face, slid to seat)

IDs are generator-assigned (annotation -> real world). Existing data/ is never touched.

Usage (from repo root):
    .venv/bin/python src/aruco_annotator/scripts/generate_aruco.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `aruco_annotator` importable

from aruco_annotator.core.aruco_placement import generate_aruco_for_object  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_MODELS = _REPO / "data" / "models"
_ASSEMBLIES = _REPO / "data"
_OUT = _REPO / "data_v2" / "aruco"
_MODEL_EXTS = (".obj", ".stl", ".ply")


def _find_model(name: str) -> Path | None:
    for ext in _MODEL_EXTS:
        candidate = _MODELS / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_object_configs() -> dict[str, dict]:
    """{name: {type, subtype}} from every assembly config (deduped; warns on conflicts)."""
    configs: dict[str, dict] = {}
    for f in sorted(_ASSEMBLIES.glob("fmb_assembly*.json")):
        data = json.loads(f.read_text())
        for comp in data.get("components", []):
            name = comp["name"]
            entry = {"type": comp.get("type", "object"), "subtype": comp.get("subtype")}
            if name in configs and configs[name] != entry:
                print(f"  WARN: {name} config differs across assemblies: {configs[name]} vs {entry}")
            configs[name] = entry
    return configs


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    configs = _load_object_configs()
    _OUT.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for name in sorted(configs):
        cfg = configs[name]
        mesh = _find_model(name)
        if mesh is None:
            print(f"  {name:18} SKIP (no model in data/models)")
            n_skip += 1
            continue
        try:
            record = generate_aruco_for_object(
                mesh, name, comp_type=cfg["type"], subtype=cfg["subtype"]
            )
        except NotImplementedError as exc:
            print(f"  {name:18} SKIP ({exc})")
            n_skip += 1
            continue
        (_OUT / f"{name}_aruco.json").write_text(json.dumps(record, indent=2))
        print(
            f"  {name:18} type={cfg['type']:6} subtype={str(cfg['subtype']):7} "
            f"-> {len(record['markers'])} markers"
        )
        n_ok += 1

    print(f"\n{n_ok} generated, {n_skip} skipped -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
