#!/usr/bin/env python3
"""
Symmetry Exporter - 3D Visualization App
FastAPI-based web application for displaying individual objects with grasp points
"""

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from shared.fastapi_utils import (
    add_cors_middleware,
    create_app,
    get_data_dir,
    run_server,
)

from .routes import components_router, symmetry_router

# Create FastAPI app using shared utility
app = create_app(
    title="Symmetry Exporter",
    description="Interactive 3D visualization tool for objects with grasp points",
    version="1.0.0",
)
add_cors_middleware(app)

# Get paths
APP_DIR = Path(__file__).parent
DATA_DIR = get_data_dir(__file__)
SHARED_DIR = APP_DIR.parent / "shared"

# Mount static files
app.mount(
    "/static/shared", StaticFiles(directory=SHARED_DIR / "static"), name="shared_static"
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Include API routes
app.include_router(components_router)
app.include_router(symmetry_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
