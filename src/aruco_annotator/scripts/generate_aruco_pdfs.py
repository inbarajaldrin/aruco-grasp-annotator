#!/usr/bin/env python3
"""
Per-object ArUco PDF export (print-and-cut sheets) into data_v2/aruco_pdfs/.

For each data_v2/aruco/{obj}_aruco.json, lays the object's markers out on an A4 page at TRUE
physical size (21 mm box/U/board, 7 mm hex), each wrapped in a black CUT-GUIDE border so the
printed marker can be trimmed to its exact `size`.

Border semantics (two distinct borders — don't confuse them):
  * white quiet zone  -> INSIDE the marker `size` (border_width fraction); part of the
    detectable rendering. The black ArUco square = size*(1-2*border_width) is what the
    aruco_camera_localizer measures for pose. (Identical to the PNG export.)
  * black cut border  -> OUTSIDE the marker, ONLY a printing artifact: a guide line to cut
    the sticker down to `size`. It is NOT part of the detectable fiducial and the localizer
    never sees it (you cut it off). PNGs omit it; PDFs include it for cutting.

Reuses ArUcoPDFGenerator (generate_aruco_pdf.py) unchanged.

Usage (from repo root):
    .venv/bin/python src/aruco_annotator/scripts/generate_aruco_pdfs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling: generate_aruco_pdf
from generate_aruco_pdf import ArUcoPDFGenerator  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ARUCO_JSON = _REPO / "data_v2" / "aruco"
_OUT = _REPO / "data_v2" / "aruco_pdfs"
_CUT_BORDER_MM = 2.0  # black cut-guide border (printing artifact, not the detected fiducial)


def main() -> int:
    if not _ARUCO_JSON.exists():
        print(f"no aruco json dir: {_ARUCO_JSON} (run generate_aruco.py first)")
        return 1

    _OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(_ARUCO_JSON.glob("*_aruco.json"))
    for json_file in files:
        name = json_file.name[: -len("_aruco.json")]
        data = json.loads(json_file.read_text())
        dict_name = data.get("aruco_dictionary", "DICT_4X4_50")
        size_mm = float(data["size"]) * 1000.0
        border_pct = float(data["border_width"]) * 100.0
        ids = [int(m["aruco_id"]) for m in data["markers"]]

        generator = ArUcoPDFGenerator(
            marker_size_mm=size_mm,
            border_width_percent=border_pct,
            page_size="A4",
            margin_mm=20,
            markers_per_row=3,
        )
        generator.generate_pdf(
            output_path=str(_OUT / f"{name}.pdf"),
            marker_ids=ids,
            dictionary=dict_name,
            add_black_border=True,            # the cut guide
            black_border_mm=_CUT_BORDER_MM,
        )
        print(f"  {name:18} {dict_name} size={size_mm:.0f}mm ids={ids} -> data_v2/aruco_pdfs/{name}.pdf")

    print(f"\n{len(files)} PDFs -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
