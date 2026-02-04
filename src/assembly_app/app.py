#!/usr/bin/env python3
"""
Enhanced Interactive 3D Assembly Web Application v2.0
FastAPI-based web application for interactive 3D component assembly with precision controls
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

from .routes import assembly_router, components_router, markers_router

# Create FastAPI app using shared utility
app = create_app(
    title="Enhanced 3D Assembly App v2.0",
    description="Interactive 3D component assembly tool with precision controls",
    version="2.0.0",
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
app.include_router(assembly_router)
app.include_router(markers_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
