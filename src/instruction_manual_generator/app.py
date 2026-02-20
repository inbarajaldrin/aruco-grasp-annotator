#!/usr/bin/env python3
"""
Instruction Manual Builder Application
Web application for creating IKEA-style assembly instruction manuals
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

from .routes import api_router

# Create FastAPI app using shared utility
app = create_app(
    title="Instruction Manual Builder",
    description="Create IKEA-style assembly instruction manuals from wireframe data",
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
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


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
