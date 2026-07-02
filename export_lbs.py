from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from global_vars import ROOT_DIR
from model.build import build_ncs_model
from epnet.data import ElasticityDataset
from epnet.topology import get_topology
from utils.IO import writePC2Frames
from utils.config import MainConfig


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_gpu(gpu_id: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")


def pc2_overwrite(path: Path, frames: np.ndarray) -> None:
    if path.is_file():
        path.unlink()
    writePC2Frames(str(path), frames)


def export_lbs(
    config: MainConfig,
    out_dir: Path,
    checkpoint_results_dir: Path | None = None,
    elasticity_iteration: int = 0,
    sequence_names: list[str] | None = None,
    warmup_frames: int = 0,
) -> None:
    print("Preparing LBS export data...")
    topology = get_topology(config)
    model = build_ncs_model(config, topology.edge_count)
    if checkpoint_results_dir is not None:
        checkpoint_path = (
            checkpoint_results_dir
            / "checkpoints"
            / config.experiment.elasticity_checkpoint
            / f"{config.name}_{elasticity_iteration}"
        )
        print("Loading elastic checkpoint for LBS weights:", checkpoint_path)
        status = model.load_weights(str(checkpoint_path))
        if hasattr(status, "expect_partial"):
            status.expect_partial()

    data = ElasticityDataset(
        config,
        topology.edge_count,
        mode="test",
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
    )
    garment_dir = out_dir / "garment"
    render_dir = out_dir / "render"
    garment_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(data):
        x, y = batch
        poses = x["poses"]
        trans = x["trans"]
        frames = y["frames"]
        if poses.ndim == 3:
            poses = poses[None]
            trans = trans[None]
            frames = frames[None]

        _, matrices = model.call_inputs(poses, trans)
        body = model.lbs_body(model.body.vertices, matrices)
        cloth_matrices = tf.gather(matrices, model.body.input_joints, axis=-3)
        garment = model.lbs_cloth(model.garment.vertices, cloth_matrices)

        name = Path(data.files_name[i]).stem
        frame_count = int(frames[0])
        export_count = frame_count + int(warmup_frames)
        body_np = np.asarray(body[0])[-export_count:]
        garment_np = np.asarray(garment[0])[-export_count:]

        np.save(garment_dir / f"{name}.npy", garment_np)
        seq_render_dir = render_dir / name
        seq_render_dir.mkdir(parents=True, exist_ok=True)
        pc2_overwrite(seq_render_dir / "body.pc2", body_np)
        pc2_overwrite(seq_render_dir / "tshirt_lbs.pc2", garment_np)
        print("Saved LBS garment:", garment_dir / f"{name}.npy")

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LBS garment motion for P-Net input experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-results-dir", default=None)
    parser.add_argument("--elasticity-iteration", type=int, default=0)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--warmup-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_gpu(args.gpu_id)
    export_lbs(
        MainConfig(str(project_path(args.config))),
        project_path(args.out_dir),
        project_path(args.checkpoint_results_dir) if args.checkpoint_results_dir else None,
        elasticity_iteration=args.elasticity_iteration,
        sequence_names=args.sequence_name,
        warmup_frames=args.warmup_frames,
    )
