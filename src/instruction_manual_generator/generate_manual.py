"""
Automated instruction manual generator.

Starts the instruction manual server, loads an assembly, configures steps
with component visibility, XYZ offsets, rotations, and camera angles,
then exports PNG and/or PDF.

Usage:
    uv run python src/instruction_manual_generator/generate_manual.py

Edit the MANUALS dict below to define your manual configurations.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# ============================================================
# MANUAL CONFIGURATIONS
# ============================================================
# Define one entry per assembly.
#
# Top-level keys:
#   title           - Manual title displayed on export
#   export          - List of formats: ["png"], ["pdf"], or ["png", "pdf"]
#   layout          - Export layout options:
#       stepsPerRow     - Columns (1, 2, or 3)
#       stepsPerPage    - Steps per page
#       resolution      - Image resolution (800, 1200, or 1600)
#   componentOverview - Component display settings for the overview section:
#       groupByType       - bool: group same-geometry components (default True)
#       useCADOrientation - bool: use original CAD orientation (default False)
#       items             - dict of component_name -> overrides:
#           displayName - Custom name shown in export (e.g. "Fork" instead of "Fork Orange")
#           count       - Number of this component (default 1)
#           hidden      - bool: exclude from component overview (default False)
#           rotation    - {x, y, z}: orientation override for component image (radians)
#   steps           - List of step definitions:
#       description - Step label text
#       size        - 1 (half width) or 2 (full width) in export layout
#       camera      - Optional saved camera angle:
#           position  - {x, y, z}: camera position
#           zoom      - Camera zoom level
#           target    - {x, y, z}: orbit target point
#       components  - dict of component_name -> per-component state:
#           visible   - bool: whether component is shown in this step
#           offset    - {x, y, z}: position offset in meters (positive Z = hover above)
#           rotation  - {x, y, z}: rotation offset in radians
#
# Component names must match the assembly JSON (e.g. "base1", "u_brown").

MANUALS = {
    "fmb_assembly1": {
        "title": "FMB Assembly 1",
        "export": ["png"],
        "layout": {"stepsPerRow": 2, "stepsPerPage": 4, "resolution": 1200},
    },
    "fmb_assembly2": {
        "title": "FMB Assembly 2",
        "export": ["png"],
        "layout": {"stepsPerRow": 2, "stepsPerPage": 4, "resolution": 1200},
    },
    "fmb_assembly3": {
        "title": "FMB Assembly 3",
        "export": ["png"],
        "layout": {"stepsPerRow": 2, "stepsPerPage": 4, "resolution": 1200},
    },
}


# ============================================================
# GENERATOR LOGIC
# ============================================================
SERVER_PORT = 8005
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "manuals"


def wait_for_server(port: int, timeout: int = 15):
    """Wait for the server to be ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/api/assemblies")
            return True
        except Exception:
            time.sleep(0.5)
    return False


def generate_manual(assembly_name: str, config: dict):
    """Generate a manual for the given assembly using headless browser."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(f"http://localhost:{SERVER_PORT}")
        page.wait_for_load_state("networkidle")

        # Wait for assembly list to be populated
        page.wait_for_function(
            f"() => document.querySelector('#assemblySelect option[value=\"{assembly_name}\"]') !== null",
            timeout=15000,
        )

        # Select and load assembly
        page.select_option("#assemblySelect", assembly_name)
        page.click("button:has-text('Load Assembly')")
        page.wait_for_function(
            "() => typeof assemblyData !== 'undefined' && assemblyData !== null",
            timeout=30000,
        )
        # Wait for OBJ models to finish loading
        time.sleep(3)

        # Set title and auto-generate steps
        title = config["title"]
        page.evaluate(f"""() => {{
            manualConfig.title = {json.dumps(title)};
            document.getElementById('manualTitle').value = {json.dumps(title)};
            autoGenerateSteps();
        }}""")

        # Set layout options
        layout = config.get("layout", {})
        if "stepsPerRow" in layout:
            page.select_option("#stepsPerRow", str(layout["stepsPerRow"]))
        if "stepsPerPage" in layout:
            page.fill("#stepsPerPage", str(layout["stepsPerPage"]))
        if "resolution" in layout:
            page.select_option("#exportResolution", str(layout["resolution"]))

        # Export
        export_formats = config.get("export", ["png"])

        if "png" in export_formats:
            print("  Exporting PNG...")
            with page.expect_download() as download_info:
                page.click("button:has-text('Export PNG')")
            download = download_info.value
            output_path = OUTPUT_DIR / f"{assembly_name}_manual.png"
            download.save_as(output_path)
            print(f"  Saved: {output_path}")

        if "pdf" in export_formats:
            print("  Exporting PDF...")
            with page.expect_download() as download_info:
                page.click("button:has-text('Export PDF')")
            download = download_info.value
            output_path = OUTPUT_DIR / f"{assembly_name}_manual.pdf"
            download.save_as(output_path)
            print(f"  Saved: {output_path}")

        browser.close()


def main():
    # Start the server
    print("Starting instruction manual server...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "instruction_manual_generator.main"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server(SERVER_PORT):
            print("ERROR: Server failed to start")
            server_proc.terminate()
            sys.exit(1)
        print(f"Server ready on port {SERVER_PORT}")

        for assembly_name, config in MANUALS.items():
            print(f"\nGenerating manual for: {assembly_name}")
            generate_manual(assembly_name, config)

        print("\nDone! All manuals generated.")
    finally:
        server_proc.terminate()
        server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
