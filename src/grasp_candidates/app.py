#!/usr/bin/env python3
"""
Grasp Candidates Visualization UI
Web-based 3D visualization tool for viewing objects with grasp points overlayed.
"""

from shared.fastapi_utils import make_app, run_server

from .routes import data_router, robot_router

app = make_app(
    title="Grasp Candidates Visualizer",
    description="Interactive 3D visualization of objects with grasp points",
    version="1.0.0",
    app_file=__file__,
    routers=[data_router, robot_router],
)


def main() -> None:
    """Main entry point for the Grasp Candidates Visualizer application."""
    run_server(
        app,
        port=8002,
        app_name="Grasp Candidates Visualizer",
        features=[
            "Visualize objects with grasp points overlayed",
            "Select objects and grasp points",
            "Interactive 3D viewer with full rotation/zoom/pan",
            "View grasp point details and positions",
        ],
    )


if __name__ == "__main__":
    main()
