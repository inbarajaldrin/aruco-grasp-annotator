"""
FastAPI utility functions shared across applications.

Provides common setup patterns for FastAPI applications including
CORS configuration, data directory resolution, and server startup.
"""

import atexit
import os
import socket
from pathlib import Path
from typing import Iterable, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


def create_app(
    title: str,
    description: str,
    version: str = "1.0.0",
) -> FastAPI:
    """
    Create a FastAPI application with standard configuration.

    Args:
        title: Application title
        description: Application description
        version: Application version (default: "1.0.0")

    Returns:
        Configured FastAPI application instance
    """
    return FastAPI(
        title=title,
        description=description,
        version=version,
    )


def add_cors_middleware(app: FastAPI) -> None:
    """
    Add CORS middleware with permissive settings for development.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_data_dir(app_file: str) -> Path:
    """
    Resolve the data directory path relative to an application file.

    The data directory is expected to be at the project root level,
    two directories up from any app module file.

    Args:
        app_file: The __file__ of the calling module (typically app.py)

    Returns:
        Path to the data directory
    """
    app_dir = Path(app_file).parent
    project_root = app_dir.parent.parent
    return project_root / "data"


def make_app(
    *,
    title: str,
    description: str,
    routers: Iterable,
    app_file: str,
    version: str = "1.0.0",
    mount_shared_static: bool = True,
) -> FastAPI:
    """
    Build a fully-wired FastAPI application.

    Absorbs the bootstrap that was duplicated across every app's ``app.py``:
    CORS, static mounts, Jinja2 templates, the standard ``index.html`` root
    handler, and router registration.

    Args:
        title: Application title.
        description: Application description.
        routers: Iterable of ``APIRouter`` to include, in order.
        app_file: The calling module's ``__file__``; used to resolve the app
            directory and its ``static``/``templates`` subdirectories.
        version: Application version (default: "1.0.0").
        mount_shared_static: Also mount ``../shared/static`` at ``/static/shared``.
            Mounted BEFORE ``/static`` so that ``/static/shared/*`` resolves to the
            shared mount (Starlette matches mounts by prefix, in order).

    Returns:
        The configured FastAPI application.
    """
    app = create_app(title=title, description=description, version=version)
    add_cors_middleware(app)

    app_dir = Path(app_file).parent
    shared_dir = app_dir.parent / "shared"

    if mount_shared_static:
        app.mount(
            "/static/shared",
            StaticFiles(directory=shared_dir / "static"),
            name="shared_static",
        )
    app.mount("/static", StaticFiles(directory=app_dir / "static"), name="static")

    templates = Jinja2Templates(directory=app_dir / "templates")

    for router in routers:
        app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        """Serve the main web interface."""
        return templates.TemplateResponse("index.html", {"request": request})

    return app


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find an available port starting from the specified port.

    Args:
        start_port: Port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            if result != 0:
                return port
    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )


def run_server(
    app: FastAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
    app_name: Optional[str] = None,
    features: Optional[list[str]] = None,
) -> None:
    """
    Run the FastAPI application with uvicorn.

    Prints startup information including URL and features.

    Args:
        app: FastAPI application instance
        host: Host address to bind to
        port: Port number to listen on
        app_name: Optional application name for display
        features: Optional list of feature descriptions to display
    """
    display_name = app_name or app.title

    print(f"Starting {display_name}...")
    print(f"Open your browser to: http://localhost:{port}")

    if features:
        print("Features:")
        for feature in features:
            print(f"   {feature}")

    print("Controls:")
    print("   Mouse wheel: zoom")
    print("   Right-click + drag: rotate")
    print("   Middle-click + drag: pan")

    # Record this running app so the agent CLI can discover it (ports are dynamic).
    # Best-effort: never let registry issues block the server. The host stored is
    # always loopback because the agent CLI only ever connects locally.
    try:
        from . import agent_registry

        agent_registry.register(display_name, port=port, host="127.0.0.1", pid=os.getpid())
        atexit.register(agent_registry.unregister, display_name)
    except Exception as exc:  # pragma: no cover - registry is non-critical
        print(f"(agent registry unavailable: {exc})")

    uvicorn.run(app, host=host, port=port)
