from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def project_cloth_outside_body(
    body_frames: np.ndarray,
    cloth_frames: np.ndarray,
    body_faces: np.ndarray,
    threshold: float,
    iterations: int = 3,
) -> np.ndarray:
    threshold = float(threshold)
    if threshold <= 0.0:
        return cloth_frames

    corrected = np.array(cloth_frames, dtype=np.float32, copy=True)
    frame_count = min(len(body_frames), len(corrected))
    for _ in range(max(1, int(iterations))):
        for frame in range(frame_count):
            body = np.asarray(body_frames[frame], dtype=np.float32)
            cloth = corrected[frame]
            nearest = cKDTree(body).query(cloth, workers=-1)[1]
            normals = vertex_normals_np(body, body_faces)
            nearest_normals = normals[nearest]
            normal_dist = np.einsum("ij,ij->i", cloth - body[nearest], nearest_normals)
            penetration = threshold - normal_dist
            mask = penetration > 0.0
            cloth[mask] += penetration[mask, None] * nearest_normals[mask]
    return corrected
