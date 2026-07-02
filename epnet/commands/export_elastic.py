from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorflow as tf

from epnet.global_vars import BODY_DIR, ROOT_DIR
from epnet.commands.train_elastic import make_output_dirs, save_elastic_threshold
from model.build import build_ncs_model
from model.cloth import Garment
from epnet.data import ElasticityDataset
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


def export_elastic(
    config: MainConfig,
    checkpoint_results_dir: Path,
    out_dir: Path,
    start_iteration: int = 0,
    end_iteration: int | None = None,
    sequence_names: list[str] | None = None,
    warmup_frames: int = 0,
    force_zero_rest: bool = False,
) -> None:
    print("Preparing elastic export data...")
    garment_obj = Path(BODY_DIR) / config.body.model / config.garment.name
    garment = Garment(str(garment_obj))
    edge_count = len(garment.edge_adjacency_index)

    print("Building elastic network...")
    elasticity_model = build_ncs_model(config, edge_count)
    test_data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
        edge_neighbors=garment.edges_neighbours,
    )

    output_dirs = make_output_dirs(out_dir)
    checkpoint_root = checkpoint_results_dir / "checkpoints" / config.experiment.elasticity_checkpoint
    stop = config.training_iteration if end_iteration is None else int(end_iteration)

    for iteration in range(int(start_iteration), stop):
        if iteration > int(start_iteration) and not force_zero_rest:
            threshold_dir = output_dirs["elasticity"] / f"iteration_{iteration - 1}" / "threshold"
            test_data.load_target(threshold_dir)

        checkpoint_path = checkpoint_root / f"{config.name}_{iteration}"
        demo_checkpoint_path = checkpoint_root / config.name
        if iteration == stop - 1 and not checkpoint_path.exists() and demo_checkpoint_path.exists():
            checkpoint_path = demo_checkpoint_path
        print("Loading elastic checkpoint:", checkpoint_path)
        status = elasticity_model.load_weights(str(checkpoint_path))
        if hasattr(status, "expect_partial"):
            status.expect_partial()

        print("Save elasticity deformation and threshold:", iteration)
        save_elastic_threshold(elasticity_model, test_data, config, iteration, output_dirs)

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EPNet elastic checkpoints without training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", required=True)
    parser.add_argument("--checkpoint-results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--end-iteration", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--force-zero-rest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_gpu(args.gpu_id)
    export_elastic(
        MainConfig(str(project_path(args.config))),
        project_path(args.checkpoint_results_dir),
        project_path(args.out_dir),
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
        sequence_names=args.sequence_name,
        warmup_frames=args.warmup_frames,
        force_zero_rest=args.force_zero_rest,
    )
