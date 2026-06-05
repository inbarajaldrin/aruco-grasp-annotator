#!/usr/bin/env python3
"""
Config-driven headless grasp-point generation into data_v2/grasp_points/.

For every graspable object in the assembly configs (type != board), runs the full grasp
pipeline (step1 render+detect -> filter -> step2 transform) and writes
{obj}_grasp_points.json. Reads aruco from data_v2/aruco (the deterministic set) and
models/wireframe from data/ (data_v2 doesn't carry those yet). Step1 render intermediates
go to a temp work dir. Boards are skipped (not grasped).

Usage (from repo root):
    .venv/bin/python src/grasp_points_annotator/scripts/generate_grasp_points.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make grasp_points_annotator importable

from grasp_points_annotator.core.grasp_pipeline import generate_grasp_points  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "data"
_DATA_V2 = _REPO / "data_v2"
_OUT = _DATA_V2 / "grasp_points"


def _object_configs() -> dict[str, str]:
    """{name: type} from the assembly configs."""
    cfg: dict[str, str] = {}
    for f in sorted(_DATA.glob("fmb_assembly*.json")):
        for comp in json.loads(f.read_text()).get("components", []):
            cfg[comp["name"]] = comp.get("type", "object")
    return cfg


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)

    # Composed read dir: aruco from data_v2, wireframe/models from data, grasp_points -> data_v2.
    staging = Path(tempfile.mkdtemp(prefix="grasp_stage_"))
    os.symlink(_DATA_V2 / "aruco", staging / "aruco")
    os.symlink(_DATA / "wireframe", staging / "wireframe")
    os.symlink(_DATA / "models", staging / "models")
    os.symlink(_OUT, staging / "grasp_points")
    work = Path(tempfile.mkdtemp(prefix="grasp_work_"))

    configs = _object_configs()
    n_ok = n_skip = n_fail = 0
    for name in sorted(configs):
        if configs[name] == "board":
            print(f"  {name:18} skip (board)")
            n_skip += 1
            continue
        if not (staging / "aruco" / f"{name}_aruco.json").exists():
            print(f"  {name:18} skip (no data_v2 aruco)")
            n_skip += 1
            continue
        try:
            out = generate_grasp_points(name, data_dir=staging, outputs_dir=work)
            d = json.loads(Path(out).read_text())
            print(
                f"  {name:18} src_marker={d['source_marker_id']} "
                f"pts={d['total_grasp_points']} markers={len(d['markers'])}"
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:18} FAIL {exc}")
            n_fail += 1

    print(f"\n{n_ok} generated, {n_skip} skipped, {n_fail} failed -> {_OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
