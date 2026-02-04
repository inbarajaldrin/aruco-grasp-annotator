#!/usr/bin/env python3
"""
Grasp Candidates Visualization UI
Web-based 3D visualization tool for viewing objects with grasp points overlayed.
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

from .routes import data_router, robot_router

# Create FastAPI app using shared utility
app = create_app(
    title="Grasp Candidates Visualizer",
    description="Interactive 3D visualization of objects with grasp points",
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
app.include_router(data_router)
app.include_router(robot_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
