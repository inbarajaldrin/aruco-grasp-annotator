#!/usr/bin/env python3
"""
Symmetry Exporter - 3D Visualization App
FastAPI-based web application for displaying individual objects with grasp points
"""

from shared.fastapi_utils import make_app, run_server

from .routes import components_router, symmetry_router

app = make_app(
    title="Symmetry Exporter",
    description="Interactive 3D visualization tool for objects with grasp points",
    version="1.0.0",
    app_file=__file__,
    routers=[components_router, symmetry_router],
)


def main() -> None:
    """Main entry point for the Symmetry Exporter application."""
    from shared.fastapi_utils import find_available_port

    port = find_available_port(8002)
    run_server(
        app,
        port=port,
        app_name="Symmetry Exporter",
        features=[
            "Load and display wireframe components",
            "Visualize grasp points",
            "Interactive 3D environment",
            "Precision position and rotation controls",
        ],
    )


if __name__ == "__main__":
    main()
