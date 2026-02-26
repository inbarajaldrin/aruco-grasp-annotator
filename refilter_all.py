#!/usr/bin/env python3
"""Batch re-filter all grasp points using the FastAPI pipeline endpoints.

Runs the full pipeline (step1 → filter → step2) for each object to regenerate
grasp point data with numeric gripper_width_mm values instead of string labels.

Usage:
    uv run python refilter_all.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be set before any other matplotlib import

import json
import time
import threading
from pathlib import Path

import requests
import uvicorn


BASE_URL = "http://127.0.0.1:8099"
DATA_DIR = Path(__file__).parent / "data" / "grasp_points"


def start_server():
    """Start the FastAPI server in a background thread."""
    from src.grasp_points_annotator.app import app
    config = uvicorn.Config(app, host="127.0.0.1", port=8099, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for server to be ready
    for _ in range(50):
        try:
            requests.get(f"{BASE_URL}/api/objects", timeout=5)
            return server
        except requests.ConnectionError:
            time.sleep(0.2)
    raise RuntimeError("Server failed to start")


def process_object(session: requests.Session, object_name: str, source_marker_id: int):
    """Run step1 → filter → step2 for a single object."""
    # Step 1: Render + detect grasp points
    r = session.post(f"{BASE_URL}/api/pipeline/step1", json={
        "object_name": object_name,
        "marker_id": source_marker_id,
    }, timeout=120)
    r.raise_for_status()
    step1 = r.json()
    print(f"  Step 1: {step1['points_3d_count']} grasp points detected")

    # Filter: Apply gripper constraints (uses new numeric output)
    r = session.post(f"{BASE_URL}/api/filter", json={
        "object_name": object_name,
        "marker_id": source_marker_id,
        "check_x_axis": True,
        "check_y_axis": True,
    }, timeout=120)
    r.raise_for_status()
    filt = r.json()
    print(f"  Filter: {filt['total_original']} → {filt['total_filtered']} grasp points")
    for res in filt["filter_results"]:
        x_w = res.get("x_axis_gripper_width_mm")
        y_w = res.get("y_axis_gripper_width_mm")
        valid = "valid" if res["is_valid"] else "INVALID"
        print(f"    grasp {res['grasp_id']}: x={x_w}mm, y={y_w}mm ({valid})")

    # Step 2: Transform to all markers → writes data/grasp_points/{object}_grasp_points.json
    r = session.post(f"{BASE_URL}/api/pipeline/step2", json={
        "object_name": object_name,
        "source_marker_id": source_marker_id,
    }, timeout=120)
    r.raise_for_status()
    step2 = r.json()
    print(f"  Step 2: {step2['total_grasp_points']} points → {step2['output_file']}")


def main():
    print("Starting FastAPI server...")
    start_server()
    print(f"Server ready at {BASE_URL}\n")

    # Discover objects from existing grasp_points JSONs
    objects = []
    for f in sorted(DATA_DIR.glob("*_grasp_points.json")):
        with open(f) as fp:
            data = json.load(fp)
        objects.append((data["object_name"], data["source_marker_id"]))

    print(f"Found {len(objects)} objects to process\n")

    session = requests.Session()
    success = 0
    failed = []

    for object_name, source_marker_id in objects:
        print(f"[{object_name}] (marker {source_marker_id})")
        try:
            process_object(session, object_name, source_marker_id)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(object_name)
        print()

    print(f"Done: {success}/{len(objects)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
