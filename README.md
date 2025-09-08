# ArUco Grasp Annotator

A 3D CAD annotation tool for defining grasp poses relative to ArUco markers on objects.

## Features

- Load 3D CAD files (STL, OBJ, PLY formats)
- Place multiple ArUco markers on 3D objects
- Define grasp poses with interactive 3D visualization
- Export annotations in JSON format
- Coordinate transformations relative to ArUco markers
- Cross-platform support (macOS, Linux, Windows)

## Installation

This project uses `uv` for fast, reliable package management.

### Prerequisites

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and setup the project:
```bash
cd "/Users/aldrininbaraj/Documents/Projects/aruco annotate"
uv sync
```

This will automatically:
- Create a virtual environment
- Install all dependencies
- Set up the development environment

## Usage

### Running the Application

```bash
# Activate the environment and run
uv run aruco-annotator

# Or manually activate and run
source .venv/bin/activate  # On macOS/Linux
python -m aruco_annotator.main
```

### Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

## Workflow

1. **Load CAD Model**: Import your STL/OBJ file
2. **Place ArUco Markers**: Position markers on the object surface
3. **Define Grasp Poses**: Use interactive 3D tools to set grasp positions and orientations
4. **Export Annotations**: Save to JSON format for use in detection pipelines

## Output Format

The tool exports annotations in JSON format with marker-relative coordinate transformations:

```json
{
  "object_file": "example.stl",
  "markers": [
    {
      "id": 0,
      "position": [x, y, z],
      "orientation": [qw, qx, qy, qz],
      "grasp_poses": [
        {
          "name": "top_grasp",
          "position": [x, y, z],
          "orientation": [qw, qx, qy, qz],
          "approach_vector": [x, y, z]
        }
      ]
    }
  ]
}
```

## Requirements

- Python 3.9+
- macOS 10.15+ / Linux / Windows 10+
- OpenGL support for 3D visualization
