from __future__ import annotations

from pathlib import Path

import numpy as np


def wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def face_normals_np(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = pos[:, faces] if pos.ndim == 3 else pos[faces]
    normals = np.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :])
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def compute_theta_np(
    pos: np.ndarray,
    faces: np.ndarray,
    face_adjacency: np.ndarray,
    hinge_vertices: np.ndarray,
) -> np.ndarray:
    face_normals = face_normals_np(pos, faces)
    if pos.ndim == 2:
        n0 = face_normals[face_adjacency[:, 0]]
        n1 = face_normals[face_adjacency[:, 1]]
        edge_vec = pos[hinge_vertices[:, 1]] - pos[hinge_vertices[:, 0]]
    else:
        n0 = face_normals[:, face_adjacency[:, 0]]
        n1 = face_normals[:, face_adjacency[:, 1]]
        edge_vec = pos[:, hinge_vertices[:, 1]] - pos[:, hinge_vertices[:, 0]]

    edge_norm = edge_vec / np.maximum(np.linalg.norm(edge_vec, axis=-1, keepdims=True), 1e-12)
    cos = np.sum(n0 * n1, axis=-1)
    sin = np.sum(edge_norm * np.cross(n0, n1), axis=-1)
    return np.arctan2(sin, cos)[..., None].astype(np.float32)


def build_hinge_graph_np(hinge_vertices: np.ndarray) -> np.ndarray:
    vertex_to_hinges: dict[int, list[int]] = {}
    for hinge_idx, (a, b) in enumerate(hinge_vertices):
        vertex_to_hinges.setdefault(int(a), []).append(hinge_idx)
        vertex_to_hinges.setdefault(int(b), []).append(hinge_idx)

    edges = set()
    for hinges in vertex_to_hinges.values():
        for i, src in enumerate(hinges):
            for dst in hinges[i + 1:]:
                edges.add((src, dst))
                edges.add((dst, src))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64).T


def hinge_midpoints_np(
    positions: np.ndarray,
    rest_positions: np.ndarray,
    hinge_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if positions.ndim == 2:
        mid_pos = 0.5 * (positions[hinge_vertices[:, 0]] + positions[hinge_vertices[:, 1]])
    else:
        mid_pos = 0.5 * (
            positions[:, hinge_vertices[:, 0]] + positions[:, hinge_vertices[:, 1]]
        )
    mid_rest = 0.5 * (
        rest_positions[hinge_vertices[:, 0]] + rest_positions[hinge_vertices[:, 1]]
    )
    return mid_pos.astype(np.float32), mid_rest.astype(np.float32)


def hinge_edge_features_np(
    mid_pos: np.ndarray,
    mid_rest: np.ndarray,
    edge_index: np.ndarray,
    material: dict,
) -> np.ndarray:
    senders = edge_index[0]
    receivers = edge_index[1]
    if mid_pos.ndim == 2:
        rel = mid_pos[senders] - mid_pos[receivers]
    else:
        rel = mid_pos[:, senders] - mid_pos[:, receivers]
    rel_norm = np.linalg.norm(rel, axis=-1, keepdims=True)

    rel_rest = mid_rest[senders] - mid_rest[receivers]
    rel_rest_norm = np.linalg.norm(rel_rest, axis=-1, keepdims=True)
    edge_count = edge_index.shape[1]
    scalars = np.array(
        [
            float(material.get("timestep", 1.0)),
            float(material.get("bending_coeff", 2.5e-4)),
            float(material.get("lame_mu", 10.0)),
            float(material.get("lame_lambda", 1.0)),
        ],
        dtype=np.float32,
    )

    if mid_pos.ndim == 2:
        return np.concatenate(
            [
                rel,
                np.maximum(rel_norm, 1e-12),
                rel_rest,
                np.maximum(rel_rest_norm, 1e-12),
                np.broadcast_to(scalars, (edge_count, 4)),
            ],
            axis=-1,
        ).astype(np.float32)

    time_count = mid_pos.shape[0]
    return np.concatenate(
        [
            rel,
            np.maximum(rel_norm, 1e-12),
            np.broadcast_to(rel_rest[None], (time_count, edge_count, 3)),
            np.broadcast_to(np.maximum(rel_rest_norm, 1e-12)[None], (time_count, edge_count, 1)),
            np.broadcast_to(scalars[None, None], (time_count, edge_count, 4)),
        ],
        axis=-1,
    ).astype(np.float32)


def prepare_sequence_features_np(
    garment_positions: np.ndarray,
    rest_positions: np.ndarray,
    faces: np.ndarray,
    face_adjacency: np.ndarray,
    hinge_vertices: np.ndarray,
    material: dict,
) -> dict[str, np.ndarray]:
    edge_index = build_hinge_graph_np(hinge_vertices)
    theta_seq = compute_theta_np(garment_positions, faces, face_adjacency, hinge_vertices)
    mid_pos_cache, mid_rest_static = hinge_midpoints_np(
        garment_positions, rest_positions, hinge_vertices
    )
    edge_feat_cache = hinge_edge_features_np(mid_pos_cache, mid_rest_static, edge_index, material)
    return {
        "theta_seq": theta_seq.astype(np.float32),
        "mid_pos_cache": mid_pos_cache.astype(np.float32),
        "edge_feat_cache": edge_feat_cache.astype(np.float32),
        "mid_rest_static": mid_rest_static.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
    }


def feature_cache_path(cache_dir: Path, sequence_name: str) -> Path:
    return Path(cache_dir) / f"{sequence_name}.npz"


def load_or_prepare_features_np(
    sequence_name: str,
    garment_positions: np.ndarray,
    topology,
    material: dict,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    if cache_dir is not None:
        path = feature_cache_path(cache_dir, sequence_name)
        if path.is_file() and not rebuild:
            data = np.load(path)
            return {key: data[key] for key in data.files}

    features = prepare_sequence_features_np(
        garment_positions,
        topology.vertices,
        topology.faces,
        topology.face_adjacency,
        topology.face_adjacency_edges,
        material,
    )
    if cache_dir is not None:
        path = feature_cache_path(cache_dir, sequence_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **features)
    return features


def load_sorted_npy(directory: Path) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p).astype(np.float32) for p in sorted(Path(directory).glob("*.npy"))}
