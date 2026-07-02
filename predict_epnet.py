from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from global_vars import CHECKPOINTS_DIR, ROOT_DIR
from model.build import build_ncs_model
from epnet.data import ElasticityDataset
from epnet.topology import get_topology
from utils import debug
from utils.collision_projection import project_cloth_outside_body
from utils.config import MainConfig
from utils.IO import writePC2Frames


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_gpu(gpu_id: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")


def save_prediction(
    elasticity_model,
    data: ElasticityDataset,
    out_root: Path,
    config: MainConfig,
    collision_projection_threshold: float | None = None,
) -> None:
    for i, batch in enumerate(data):
        body, garment, unskinned, plasticity_target, frames = elasticity_model.predict(batch, w=1.0)
        name, _ = os.path.splitext(data.files_name[i])
        frame_count = int(frames[0])
        export_count = frame_count + int(data.warmup_frames)

        print("File name: [", name, "] Frame count: ", frame_count)
        if data.warmup_frames > 0:
            print("Export warmup frames:", int(data.warmup_frames))
        angle = debug.deformation_to_signed_angle(garment, elasticity_model.garment)
        angle_delta = angle - elasticity_model.garment.face_dir_dihedral
        angle_delta = tf.math.atan2(tf.sin(angle_delta), tf.cos(angle_delta))
        angle_delta = angle_delta[:, -export_count:]
        threshold = debug.signed_angle_to_threshold(
            angle_delta[..., None],
            config.step_smooth_scale,
            config.angle_step,
        )

        render_path = out_root / "render" / name
        render_path.mkdir(parents=True, exist_ok=True)

        body = np.array(body[0])[-export_count:]
        garment = np.array(garment[0])[-export_count:]
        unskinned = np.array(unskinned[0])[-export_count:]
        threshold = np.array(threshold[0, :, :, 0])[-export_count:]
        projection_threshold = (
            float(getattr(config.loss, "collision_projection_threshold", 0.0))
            if collision_projection_threshold is None
            else float(collision_projection_threshold)
        )
        garment = project_cloth_outside_body(
            body,
            garment,
            elasticity_model.body.faces,
            projection_threshold,
        )

        np.save(render_path / f"{name}_garment.npy", garment)
        np.save(render_path / f"{name}_threshold.npy", threshold)

        pc2_outputs = {
            "body.pc2": body,
            "tshirt.pc2": garment,
            "tshirt_unskinned.pc2": unskinned,
        }
        for filename, frames_out in pc2_outputs.items():
            path = render_path / filename
            if path.exists():
                path.unlink()
            writePC2Frames(str(path), frames_out)


def main(
    config: MainConfig,
    plasticity_dir: Path,
    out_dir: Path | None = None,
    max_sequences: int | None = None,
    sequence_names: list[str] | None = None,
    results_dir: Path | None = None,
    warmup_frames: int = 0,
    elasticity_iteration: int | None = None,
    collision_projection_threshold: float | None = None,
) -> None:
    print("Preparing EPNet prediction data...")
    topology = get_topology(config)
    edge_count = topology.edge_count

    print("Building elastic network...")
    elasticity_model = build_ncs_model(config, edge_count)
    test_data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        max_sequences=max_sequences,
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
    )
    test_data.load_target(plasticity_dir)

    checkpoint_root = (results_dir / "checkpoints") if results_dir else Path(CHECKPOINTS_DIR)
    iteration = config.training_iteration - 1 if elasticity_iteration is None else int(elasticity_iteration)
    checkpoint_path = checkpoint_root / config.experiment.elasticity_checkpoint / f"{config.name}_{iteration}"
    print("Loading elastic checkpoint:", checkpoint_path)
    status = elasticity_model.load_weights(str(checkpoint_path))
    if hasattr(status, "expect_partial"):
        status.expect_partial()

    if out_dir:
        out_root = out_dir
    elif results_dir:
        out_root = results_dir / "prediction_epnet"
    else:
        out_root = Path(ROOT_DIR) / "results" / "prediction_epnet"
    print("Start EPNet predicting...")
    save_prediction(
        elasticity_model,
        test_data,
        out_root,
        config,
        collision_projection_threshold=collision_projection_threshold,
    )
    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with EPNet elastic model using external P-Net plasticity files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--plasticity-dir", default=r"results\rest")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--elasticity-iteration", type=int, default=None)
    parser.add_argument("--collision-projection-threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_gpu(args.gpu_id)
    main(
        MainConfig(str(project_path(args.config))),
        project_path(args.plasticity_dir),
        project_path(args.out_dir) if args.out_dir else None,
        max_sequences=args.max_sequences,
        sequence_names=args.sequence_name,
        results_dir=project_path(args.results_dir) if args.results_dir else None,
        warmup_frames=args.warmup_frames,
        elasticity_iteration=args.elasticity_iteration,
        collision_projection_threshold=args.collision_projection_threshold,
    )
