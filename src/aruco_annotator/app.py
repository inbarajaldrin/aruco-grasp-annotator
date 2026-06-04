#!/usr/bin/env python3
"""
ArUco Grasp Annotator Web Application
FastAPI-based web application for placing ArUco markers on 3D CAD objects
"""

from shared.fastapi_utils import make_app, run_server

from .routes import export_router, markers_router, model_router, placement_router

app = make_app(
    title="ArUco Grasp Annotator",
    description="3D CAD annotation tool for placing ArUco markers on objects",
    version="2.0.0",
    app_file=__file__,
    routers=[model_router, markers_router, placement_router, export_router],
)


def main() -> None:
    """Main entry point for the ArUco Grasp Annotator application."""
    from shared.fastapi_utils import find_available_port

    port = find_available_port(8000)
    run_server(
        app,
        port=port,
        app_name="ArUco Grasp Annotator Web App",
        features=[
            "Load and display CAD models (STL, OBJ, PLY)",
            "Place ArUco markers with multiple placement modes",
            "Interactive 3D visualization",
            "Export/import annotations",
        ],
    )


if __name__ == "__main__":
    main()
