"""
Shared modules for ArUco Grasp Annotator applications.

This package contains common utilities, components, and configurations
that are reused across multiple applications in the project.
"""

from .fastapi_utils import (
    create_app,
    add_cors_middleware,
    get_data_dir,
    find_available_port,
    run_server,
)
from .data_loaders import (
    get_available_components,
    get_available_objects,
    load_wireframe_data,
    load_aruco_data,
    load_grasp_data,
    transform_aruco_data,
)
from .aruco_utils import (
    ARUCO_DICTS,
    generate_aruco_marker,
)

__all__ = [
    # FastAPI utilities
    "create_app",
    "add_cors_middleware",
    "get_data_dir",
    "find_available_port",
    "run_server",
    # Data loaders
    "get_available_components",
    "get_available_objects",
    "load_wireframe_data",
    "load_aruco_data",
    "load_grasp_data",
    "transform_aruco_data",
    # ArUco utilities
    "ARUCO_DICTS",
    "generate_aruco_marker",
]
