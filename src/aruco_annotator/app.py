#!/usr/bin/env python3
"""
ArUco Grasp Annotator Web Application
FastAPI-based web application for placing ArUco markers on 3D CAD objects
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

from .routes import export_router, markers_router, model_router, placement_router

# Create FastAPI app using shared utility
app = create_app(
    title="ArUco Grasp Annotator",
    description="3D CAD annotation tool for placing ArUco markers on objects",
    version="2.0.0",
)
add_cors_middleware(app)

# Get paths
APP_DIR = Path(__file__).parent
DATA_DIR = get_data_dir(__file__)
SHARED_DIR = APP_DIR.parent / "shared"

# Mount static files
app.mount("/static/shared", StaticFiles(directory=SHARED_DIR / "static"), name="shared_static")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Include API routes
app.include_router(model_router)
app.include_router(markers_router)
app.include_router(placement_router)
app.include_router(export_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
