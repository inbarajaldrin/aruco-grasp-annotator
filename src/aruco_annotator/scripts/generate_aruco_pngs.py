#!/usr/bin/env python3
"""
Per-object ArUco marker PNG export into data_v2/aruco_pngs/{object}/.

Reuses the existing ArUcoPNGGenerator (generate_aruco_png.py) unchanged — the rendering the
downstream aruco_camera_localizer expects: white quiet zone INSIDE, black detectable square =
size·(1 − 2·border). For each object's data_v2/aruco/{name}_aruco.json it reads the
dictionary, size, border_width, and marker ids, and renders one PNG per marker at the
object's own size (21 mm box/U/board, 7 mm hex). Defaults (dpi 304.8, no outer black frame)
reproduce the committed data/aruco/pngs markers byte-for-byte.

NOTE: marker size is per-object. The localizer currently assumes 21 mm for every marker
(documented future TODO) — a 7 mm hex marker only detects correctly once the localizer reads
`size` from the JSON.

Usage (from repo root):
    .venv/bin/python src/aruco_annotator/scripts/generate_aruco_pngs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling: generate_aruco_png
from generate_aruco_png import ArUcoPNGGenerator  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ARUCO_JSON = _REPO / "data_v2" / "aruco"
_OUT = _REPO / "data_v2" / "aruco_pngs"
_DPI = 304.8  # matches the maintained rendering


def main() -> int:
    if not _ARUCO_JSON.exists():
        print(f"no aruco json dir: {_ARUCO_JSON} (run generate_aruco.py first)")
        return 1

    files = sorted(_ARUCO_JSON.glob("*_aruco.json"))
    total = 0
    for json_file in files:
        name = json_file.name[: -len("_aruco.json")]
        data = json.loads(json_file.read_text())
        dict_name = data.get("aruco_dictionary", "DICT_4X4_50")
        size_mm = float(data["size"]) * 1000.0
        border_pct = float(data["border_width"]) * 100.0

        generator = ArUcoPNGGenerator(
            marker_size_mm=size_mm, border_width_percent=border_pct, dpi=_DPI
        )
        out_dir = _OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        ids = []
        for marker in data["markers"]:
            mid = int(marker["aruco_id"])
            img = generator.generate_marker_image(mid, dict_name, add_black_border=False)
            cv2.imwrite(str(out_dir / f"{name}_marker_{mid:03d}.png"), img)
            ids.append(mid)

        total += len(ids)
        print(f"  {name:18} {dict_name} size={size_mm:.0f}mm border={border_pct:.0f}% ids={ids}")

    print(f"\n{total} PNGs -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
