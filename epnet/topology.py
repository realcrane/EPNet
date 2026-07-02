from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epnet.global_vars import BODY_DIR, ROOT_DIR


@dataclass
class GarmentTopology:
    vertices: np.ndarray
    faces: np.ndarray
    face_adjacency: np.ndarray
    face_adjacency_edges: np.ndarray
    edge_count: int


def default_topology_cache(config) -> Path:
    garment_name = Path(config.garment.name).stem
    return Path(ROOT_DIR) / "cache" / "topology" / f"{config.body.model}_{garment_name}.npz"


def load_topology(path: Path) -> GarmentTopology:
    data = np.load(path)
    face_adjacency_edges = data["face_adjacency_edges"].astype(np.int64)
    return GarmentTopology(
        vertices=data["vertices"].astype(np.float32),
        faces=data["faces"].astype(np.int64),
        face_adjacency=data["face_adjacency"].astype(np.int64),
        face_adjacency_edges=face_adjacency_edges,
        edge_count=int(data["edge_count"]) if "edge_count" in data.files else int(len(face_adjacency_edges)),
    )


def save_topology(path: Path, topology: GarmentTopology) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=topology.vertices,
        faces=topology.faces,
        face_adjacency=topology.face_adjacency,
        face_adjacency_edges=topology.face_adjacency_edges,
        edge_count=np.array(topology.edge_count, dtype=np.int64),
    )


def build_topology(config) -> GarmentTopology:
    from model.cloth import Garment

    garment_path = Path(BODY_DIR) / config.body.model / config.garment.name
    garment = Garment(str(garment_path))
    return GarmentTopology(
        vertices=garment.vertices.astype(np.float32),
        faces=garment.faces.astype(np.int64),
        face_adjacency=garment.face_adjacency.astype(np.int64),
        face_adjacency_edges=garment.face_adjacency_edges.astype(np.int64),
        edge_count=int(len(garment.edge_adjacency_index)),
    )


def get_topology(config, cache_path: Path | None = None, rebuild: bool = False) -> GarmentTopology:
    cache_path = cache_path or default_topology_cache(config)
    if cache_path.is_file() and not rebuild:
        return load_topology(cache_path)
    topology = build_topology(config)
    save_topology(cache_path, topology)
    return topology
