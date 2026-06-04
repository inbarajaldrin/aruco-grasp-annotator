#!/usr/bin/env python3
"""
Instruction Manual Builder Application
Web application for creating IKEA-style assembly instruction manuals
"""

from shared.fastapi_utils import make_app, run_server

from .routes import api_router

app = make_app(
    title="Instruction Manual Builder",
    description="Create IKEA-style assembly instruction manuals from wireframe data",
    version="1.0.0",
    app_file=__file__,
    routers=[api_router],
)


def main() -> None:
    """Main entry point for the Instruction Manual Builder application."""
    from shared.fastapi_utils import find_available_port

    port = find_available_port(8005)
    run_server(
        app,
        port=port,
        app_name="Instruction Manual Builder",
        features=[
            "Load assembly configurations",
            "Build step-by-step instructions",
            "Toggle component visibility per step",
            "Adjust component offsets for exploded views",
            "Export as PNG and PDF",
            "Save/load manual configurations",
        ],
    )


if __name__ == "__main__":
    main()
