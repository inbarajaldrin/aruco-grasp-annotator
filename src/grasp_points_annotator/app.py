#!/usr/bin/env python3
"""
Grasp Points Annotation Web Application
FastAPI-based web application for generating grasp points from CAD models
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

from .routes import filter_router, objects_router, pipeline_router, results_router

# Create FastAPI app using shared utility
app = create_app(
    title="Grasp Points Annotation App",
    description="Interactive web application for generating grasp points from CAD models",
    version="1.0.0",
)
add_cors_middleware(app)

# Get paths
APP_DIR = Path(__file__).parent
DATA_DIR = get_data_dir(__file__)
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Debug: Print paths on startup
print(f"Data directory: {DATA_DIR}")
print(f"Data directory exists: {DATA_DIR.exists()}")
print(f"Outputs directory: {OUTPUTS_DIR}")

# Mount static files
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Include API routes
app.include_router(objects_router)
app.include_router(pipeline_router)
app.include_router(filter_router)
app.include_router(results_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main application interface."""
    return templates.TemplateResponse("index.html", {"request": request})


def main() -> None:
    """Main entry point for the grasp points annotation application."""
    from shared.fastapi_utils import find_available_port

    port = find_available_port(8002)
    run_server(
        app,
        port=port,
        app_name="Grasp Points Annotation Web App",
        features=[
            "Load CAD models and ArUco annotations",
            "Generate grasp points from top-down views",
            "Transform grasp points to all markers",
            "Export results for robotics pipelines",
        ],
    )


if __name__ == "__main__":
    main()
