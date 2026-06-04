#!/usr/bin/env python3
"""
Grasp Points Annotation Web Application
FastAPI-based web application for generating grasp points from CAD models
"""

from pathlib import Path

from shared.fastapi_utils import get_data_dir, make_app, run_server

from .routes import filter_router, objects_router, pipeline_router, results_router

app = make_app(
    title="Grasp Points Annotation App",
    description="Interactive web application for generating grasp points from CAD models",
    version="1.0.0",
    app_file=__file__,
    routers=[objects_router, pipeline_router, filter_router, results_router],
    mount_shared_static=False,
)

# App-specific paths and startup diagnostics
APP_DIR = Path(__file__).parent
DATA_DIR = get_data_dir(__file__)
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Data directory exists: {DATA_DIR.exists()}")
print(f"Outputs directory: {OUTPUTS_DIR}")


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
