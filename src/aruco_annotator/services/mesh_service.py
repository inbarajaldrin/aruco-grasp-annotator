"""Mesh processing and face detection services."""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def group_triangles_by_face(
    triangles: np.ndarray, vertices: np.ndarray
) -> list[tuple]:
    """
    Face detection algorithm that groups triangles into coplanar faces.

    Args:
        triangles: Array of triangle vertex indices
        vertices: Array of vertex coordinates

    Returns:
        List of tuples: (face_center, face_normal, total_area, triangle_indices)
    """
    face_groups = []
    normal_tolerance = 0.02
    spatial_threshold = 0.1
    min_face_area = 1e-6

    triangle_data = []
    for i, triangle in enumerate(triangles):
        v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(normal)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        center = (v1 + v2 + v3) / 3.0
        plane_d = -np.dot(normal, center)

        triangle_data.append({
            "vertices": (v1, v2, v3),
            "center": center,
            "normal": normal,
            "area": area,
            "plane_d": plane_d,
            "triangle_idx": i,
        })

    used_triangles = set()
    sorted_indices = sorted(
        range(len(triangle_data)), key=lambda i: triangle_data[i]["area"], reverse=True
    )

    for idx in sorted_indices:
        if idx in used_triangles:
            continue

        seed_triangle = triangle_data[idx]
        if seed_triangle["area"] < min_face_area:
            continue

        face_triangles = [idx]
        face_queue = [idx]
        used_triangles.add(idx)

        while face_queue:
            current_idx = face_queue.pop(0)
            current_triangle = triangle_data[current_idx]

            for j, candidate in enumerate(triangle_data):
                if j in used_triangles:
                    continue

                normal_similarity = np.dot(
                    current_triangle["normal"], candidate["normal"]
                )
                if normal_similarity < (1.0 - normal_tolerance):
                    continue

                point_to_plane_dist = abs(
                    np.dot(candidate["normal"], current_triangle["center"])
                    + current_triangle["plane_d"]
                )
                if point_to_plane_dist > spatial_threshold * 0.1:
                    continue

                is_spatially_connected = False
                for face_tri_idx in face_triangles:
                    face_center = triangle_data[face_tri_idx]["center"]
                    distance = np.linalg.norm(candidate["center"] - face_center)
                    if distance < spatial_threshold:
                        is_spatially_connected = True
                        break

                if not is_spatially_connected:
                    continue

                tri1_verts = set(triangles[current_idx])
                tri2_verts = set(triangles[j])
                shared_vertices = tri1_verts.intersection(tri2_verts)
                is_edge_connected = len(shared_vertices) >= 1

                if is_edge_connected:
                    face_triangles.append(j)
                    face_queue.append(j)
                    used_triangles.add(j)

        if len(face_triangles) > 0:
            face_centers = [triangle_data[i]["center"] for i in face_triangles]
            face_areas = [triangle_data[i]["area"] for i in face_triangles]
            face_normals = [triangle_data[i]["normal"] for i in face_triangles]

            total_area = sum(face_areas)
            if total_area > min_face_area:
                face_center = np.average(face_centers, axis=0, weights=face_areas)
                face_normal = np.average(face_normals, axis=0, weights=face_areas)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)

                face_groups.append(
                    (face_center, face_normal, total_area, face_triangles)
                )

    face_groups.sort(key=lambda x: x[2], reverse=True)
    return face_groups


def calculate_rotation_from_normal(normal: tuple) -> tuple:
    """Calculate RPY rotation from surface normal."""
    n = np.array(normal)
    n = n / (np.linalg.norm(n) + 1e-8)

    z_axis = np.array([0.0, 0.0, 1.0])

    if np.allclose(n, z_axis, atol=1e-6):
        return (0.0, 0.0, 0.0)
    if np.allclose(n, -z_axis, atol=1e-6):
        return (np.pi, 0.0, 0.0)

    if np.allclose(n, [1, 0, 0], atol=1e-6):
        return (0.0, np.pi / 2, 0.0)
    elif np.allclose(n, [-1, 0, 0], atol=1e-6):
        return (0.0, -np.pi / 2, 0.0)
    elif np.allclose(n, [0, 1, 0], atol=1e-6):
        return (-np.pi / 2, 0.0, 0.0)
    elif np.allclose(n, [0, -1, 0], atol=1e-6):
        return (np.pi / 2, 0.0, 0.0)
    elif np.allclose(n, [0, 0, -1], atol=1e-6):
        return (np.pi, 0.0, 0.0)
    else:
        rotation_axis = np.cross(z_axis, n)
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, n), -1.0, 1.0))

        if abs(rotation_axis[2]) > 0.9:
            return (
                0.0,
                0.0,
                rotation_angle if rotation_axis[2] > 0 else -rotation_angle,
            )
        elif abs(rotation_axis[1]) > 0.9:
            return (
                0.0,
                rotation_angle if rotation_axis[1] > 0 else -rotation_angle,
                0.0,
            )
        else:
            return (
                rotation_angle if rotation_axis[0] > 0 else -rotation_angle,
                0.0,
                0.0,
            )


def determine_face_type(normal: tuple) -> str:
    """Determine face type from normal vector."""
    n = np.array(normal)
    n = n / (np.linalg.norm(n) + 1e-8)

    if np.allclose(n, [0, 0, 1], atol=0.1):
        return "top"
    elif np.allclose(n, [0, 0, -1], atol=0.1):
        return "bottom"
    elif np.allclose(n, [0, 1, 0], atol=0.1):
        return "front"
    elif np.allclose(n, [0, -1, 0], atol=0.1):
        return "back"
    elif np.allclose(n, [1, 0, 0], atol=0.1):
        return "right"
    elif np.allclose(n, [-1, 0, 0], atol=0.1):
        return "left"
    else:
        return "custom"


def find_closest_surface_point_and_normal(
    target_point: np.ndarray, mesh: Any, triangles: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest point on mesh surface and its normal."""
    closest_distance = float("inf")
    closest_point = None
    closest_normal = None

    for triangle in triangles:
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        tri_normal = np.cross(v1 - v0, v2 - v0)
        tri_length = np.linalg.norm(tri_normal)
        if tri_length < 1e-8:
            continue
        tri_normal = tri_normal / tri_length

        edge1 = v1 - v0
        edge2 = v2 - v0
        v0_to_p = target_point - v0

        dot00 = np.dot(edge1, edge1)
        dot01 = np.dot(edge1, edge2)
        dot02 = np.dot(edge1, v0_to_p)
        dot11 = np.dot(edge2, edge2)
        dot12 = np.dot(edge2, v0_to_p)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        u = max(0, min(1, u))
        v = max(0, min(1, v))
        if u + v > 1:
            u = 1 - v
            v = 1 - u

        triangle_point = v0 + u * edge1 + v * edge2
        distance = np.linalg.norm(triangle_point - target_point)

        if distance < closest_distance:
            closest_distance = distance
            closest_point = triangle_point
            closest_normal = tri_normal

    if closest_point is None:
        vertex_distances = np.linalg.norm(vertices - target_point, axis=1)
        closest_idx = np.argmin(vertex_distances)
        closest_point = vertices[closest_idx]
        if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > closest_idx:
            closest_normal = np.array(mesh.vertex_normals[closest_idx])
        else:
            closest_normal = np.array([0.0, 0.0, 1.0])

    return closest_point, closest_normal


def project_to_surface_with_raycast(
    start_point: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    vertices: np.ndarray,
    max_distance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a point onto the mesh surface along a direction vector."""
    direction = np.array(direction)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    closest_intersection = None
    closest_distance = float("inf")
    closest_normal = None

    for triangle in triangles:
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        tri_normal = np.cross(edge1, edge2)
        tri_length = np.linalg.norm(tri_normal)
        if tri_length < 1e-8:
            continue
        tri_normal = tri_normal / tri_length

        denom = np.dot(direction, tri_normal)
        if abs(denom) < 1e-8:
            continue

        t = np.dot(v0 - start_point, tri_normal) / denom
        if t < 0 or t > max_distance:
            continue

        intersection = start_point + t * direction

        v0_to_p = intersection - v0
        dot00 = np.dot(edge1, edge1)
        dot01 = np.dot(edge1, edge2)
        dot02 = np.dot(edge1, v0_to_p)
        dot11 = np.dot(edge2, edge2)
        dot12 = np.dot(edge2, v0_to_p)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        if u >= 0 and v >= 0 and u + v <= 1:
            if t < closest_distance:
                closest_distance = t
                closest_intersection = intersection
                closest_normal = tri_normal

    if closest_intersection is not None:
        return closest_intersection, closest_normal

    class DummyMesh:
        def __init__(self, triangles, vertices):
            self.triangles = triangles
            self.vertices = vertices

    dummy_mesh = DummyMesh(triangles, vertices)
    return find_closest_surface_point_and_normal(
        start_point, dummy_mesh, triangles, vertices
    )
