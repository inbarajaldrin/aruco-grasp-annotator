#!/usr/bin/env python3
"""
Enhanced Interactive 3D Assembly Web Application v2.0
FastAPI-based web application for interactive 3D component assembly with precision controls
"""

from shared.fastapi_utils import make_app, run_server

from .routes import assembly_router, components_router, markers_router

app = make_app(
    title="Enhanced 3D Assembly App v2.0",
    description="Interactive 3D component assembly tool with precision controls",
    version="2.0.0",
    app_file=__file__,
    routers=[components_router, assembly_router, markers_router],
)


def main() -> None:
    """Main entry point for the Enhanced Assembly application."""
    from shared.fastapi_utils import find_available_port

    port = find_available_port(8001)
    run_server(
        app,
        port=port,
        app_name="Enhanced 3D Assembly App v2.0",
        features=[
            "Load and display wireframe components",
            "Precision position and rotation controls",
            "ArUco marker visualization",
            "Interactive assembly with selection",
            "Export assembly configurations",
        ],
    )


if __name__ == "__main__":
    main()
