#!/usr/bin/env python3
"""
Regenerate the existing data/symmetry/*_symmetry.json files from CAD geometry.

Replaces the previously human-annotated fold symmetry with the deterministic auto-detector,
proving the autoload reproduces the ground truth. Keys off the EXISTING symmetry files, so it
only touches objects that already have one (base1/2/3 stay untouched by design).

Usage (from the repo root):
    .venv/bin/python src/symmetry_exporter/scripts/regen_symmetry.py [--check]

--check writes nothing; just reports whether each regenerated record matches the file on disk.
After a real run, `git diff --stat data/symmetry/` should be empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the `symmetry_exporter` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from symmetry_exporter.core.fold_axes import auto_detect_fold_axes  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "data"
_MODEL_EXTS = (".obj", ".stl", ".ply")


def _find_model(object_name: str) -> Path | None:
    for ext in _MODEL_EXTS:
        candidate = _DATA / "models" / f"{object_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    sym_dir = _DATA / "symmetry"
    existing = sorted(sym_dir.glob("*_symmetry.json"))
    if not existing:
        print(f"no existing symmetry files under {sym_dir}")
        return 1

    n_match = n_changed = n_error = 0
    for sym_file in existing:
        name = sym_file.name[: -len("_symmetry.json")]
        mesh = _find_model(name)
        if mesh is None:
            print(f"  {name:20} SKIP (no model)")
            n_error += 1
            continue
        try:
            record = auto_detect_fold_axes(mesh, name)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:20} ERROR {e}")
            n_error += 1
            continue

        new_text = json.dumps(record, indent=4)
        old_text = sym_file.read_text()
        if new_text == old_text:
            print(f"  {name:20} unchanged (byte-identical)")
            n_match += 1
        else:
            folds = {a: v["fold"] for a, v in record["fold_axes"].items() if v["fold"] >= 2}
            print(f"  {name:20} {'WOULD CHANGE' if args.check else 'rewritten'}  folds={folds}")
            if not args.check:
                sym_file.write_text(new_text)
            n_changed += 1

    print(f"\n{n_match} identical, {n_changed} changed, {n_error} error(s) "
          f"({'check-only' if args.check else 'written'})")
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
