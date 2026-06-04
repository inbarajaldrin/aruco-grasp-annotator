"""
Wireframe extraction from a loaded CAD mesh.

Pulls the vertices + unique triangle edges + mesh_info out of an open3d TriangleMesh
into the schema used by ``data/wireframe/{object}_wireframe.json``. Shared by the
interactive ``/api/export-wireframe`` (uses the UI-loaded mesh) and the headless
``/api/export-wireframe/{object_name}`` (loads by name) so both emit identical output.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_wireframe(mesh) -> dict[str, Any]:
    """Build the wireframe record (vertices, unique edges, mesh_info) for an open3d mesh.

    Edges are the de-duplicated, vertex-sorted triangle edges, kept in first-seen order.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    seen: set[tuple[int, int]] = set()
    unique_edges: list[tuple[int, int]] = []
    for triangle in triangles:
        for i in range(3):
            v1, v2 = triangle[i], triangle[(i + 1) % 3]
            edge = tuple(sorted([int(v1), int(v2)]))
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)

    mesh_info = {
        "num_vertices": len(vertices),
        "num_edges": len(unique_edges),
        "num_triangles": len(triangles),
        "bounding_box": {
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
            "center": vertices.mean(axis=0).tolist(),
            "size": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
        },
        "has_normals": mesh.has_vertex_normals(),
        "has_colors": mesh.has_vertex_colors(),
        "is_watertight": mesh.is_watertight(),
        "is_orientable": mesh.is_orientable(),
    }

    return {
        "mesh_info": mesh_info,
        "vertices": vertices.tolist(),
        "edges": [[int(edge[0]), int(edge[1])] for edge in unique_edges],
        "format": "vector_relation",
        "description": "Wireframe data with vertices and edge connections",
    }
