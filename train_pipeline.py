from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from global_vars import ROOT_DIR
from utils.config import MainConfig


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_step(args: list[str]) -> None:
    print("[pipeline]", " ".join(str(arg) for arg in args))
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def train_elastic_iteration(
    config: Path,
    gpu_id: str,
    iteration: int,
    target_dir: Path | None,
    sequence_names: list[str] | None,
    results_dir: Path,
    elasticity_epochs: int | None,
    elasticity_initial_epochs: int | None,
    elasticity_later_epochs: int | None,
    target_input_noise_std: float,
    target_noise_as_target: bool,
    target_noise_smoothing_steps: int,
    warmup_frames: int,
    enable_tensorboard: bool,
    checkpoint_each_epoch: bool,
    rest_zero_rehearsal_probability: float,
) -> None:
    args = [
        sys.executable,
        "main.py",
        "--config",
        str(config),
        "--gpu_id",
        str(gpu_id),
        "--start-iteration",
        str(iteration),
        "--end-iteration",
        str(iteration + 1),
        "--results-dir",
        str(results_dir),
    ]
    if elasticity_epochs is not None:
        args.extend(["--elasticity-epochs", str(elasticity_epochs)])
    if elasticity_initial_epochs is not None:
        args.extend(["--elasticity-initial-epochs", str(elasticity_initial_epochs)])
    if elasticity_later_epochs is not None:
        args.extend(["--elasticity-later-epochs", str(elasticity_later_epochs)])
    if target_input_noise_std > 0:
        args.extend(["--target-input-noise-std", str(target_input_noise_std)])
    if target_noise_as_target:
        args.append("--target-noise-as-target")
    if target_noise_smoothing_steps > 0:
        args.extend(["--target-noise-smoothing-steps", str(target_noise_smoothing_steps)])
    if warmup_frames > 0:
        args.extend(["--warmup-frames", str(warmup_frames)])
    if enable_tensorboard:
        args.append("--tensorboard")
    if checkpoint_each_epoch:
        args.append("--checkpoint-each-epoch")
    if rest_zero_rehearsal_probability > 0:
        args.extend(["--rest-zero-rehearsal-probability", str(rest_zero_rehearsal_probability)])
    if target_dir is not None:
        args.extend(["--target-dir", str(target_dir)])
    if sequence_names:
        for name in sequence_names:
            args.extend(["--sequence-name", name])
    run_step(args)


def train_pnet_iteration(
    config: Path,
    results_dir: Path,
    iteration: int,
    epochs: int,
    time_batch_size: int,
    active_weight: float,
    active_threshold: float,
    inactive_weight: float,
    feature_cache_dir: Path | None,
    topology_cache: Path | None,
    sequence_names: list[str] | None,
) -> Path:
    garment_dir = results_dir / "elasticity" / f"iteration_{iteration}" / "garment"
    target_dir = results_dir / "elasticity" / f"iteration_{iteration}" / "threshold"
    checkpoint = results_dir / "checkpoints" / "pnet" / f"iteration_{iteration}" / "pnet.weights.h5"

    args = [
        sys.executable,
        "train_pnet_tf.py",
        "--config",
        str(config),
        "--garment-dir",
        str(garment_dir),
        "--target-dir",
        str(target_dir),
        "--epochs",
        str(epochs),
        "--time-batch-size",
        str(time_batch_size),
        "--active-weight",
        str(active_weight),
        "--active-threshold",
        str(active_threshold),
        "--inactive-weight",
        str(inactive_weight),
        "--out",
        str(checkpoint),
    ]
    if feature_cache_dir is not None:
        args.extend(["--feature-cache-dir", str(feature_cache_dir / f"iteration_{iteration}")])
    if topology_cache is not None:
        args.extend(["--topology-cache", str(topology_cache)])
    if sequence_names:
        for name in sequence_names:
            args.extend(["--sequence-name", name])
    run_step(args)
    return checkpoint


def predict_pnet_iteration(
    config: Path,
    results_dir: Path,
    iteration: int,
    checkpoint: Path,
    alpha_cutoff: float,
    feature_cache_dir: Path | None,
    topology_cache: Path | None,
    sequence_names: list[str] | None,
) -> Path:
    garment_dir = results_dir / "elasticity" / f"iteration_{iteration}" / "garment"
    out_dir = results_dir / "rest" / f"iteration_{iteration}"
    alpha_dir = results_dir / "alpha" / f"iteration_{iteration}"

    args = [
        sys.executable,
        "predict_pnet_tf.py",
        "--config",
        str(config),
        "--ckpt",
        str(checkpoint),
        "--garment-dir",
        str(garment_dir),
        "--out-dir",
        str(out_dir),
        "--alpha-dir",
        str(alpha_dir),
    ]
    if alpha_cutoff > 0.0:
        args.extend(["--alpha-cutoff", str(alpha_cutoff)])
    if feature_cache_dir is not None:
        args.extend(["--feature-cache-dir", str(feature_cache_dir / f"iteration_{iteration}")])
    if topology_cache is not None:
        args.extend(["--topology-cache", str(topology_cache)])
    if sequence_names:
        for name in sequence_names:
            args.extend(["--sequence-name", name])
    run_step(args)
    return out_dir


def run_debug(args: argparse.Namespace, config: Path) -> None:
    elastic_args = [
        sys.executable,
        "main.py",
        "--config",
        str(config),
        "--gpu_id",
        str(args.gpu_id),
        "--results-dir",
        str(args.results_dir),
    ]
    if args.elasticity_epochs is not None:
        elastic_args.extend(["--elasticity-epochs", str(args.elasticity_epochs)])
    if args.elasticity_initial_epochs is not None:
        elastic_args.extend(["--elasticity-initial-epochs", str(args.elasticity_initial_epochs)])
    if args.elasticity_later_epochs is not None:
        elastic_args.extend(["--elasticity-later-epochs", str(args.elasticity_later_epochs)])
    if args.target_input_noise_std > 0:
        elastic_args.extend(["--target-input-noise-std", str(args.target_input_noise_std)])
    if args.target_noise_as_target:
        elastic_args.append("--target-noise-as-target")
    if args.target_noise_smoothing_steps > 0:
        elastic_args.extend(["--target-noise-smoothing-steps", str(args.target_noise_smoothing_steps)])
    if args.warmup_frames > 0:
        elastic_args.extend(["--warmup-frames", str(args.warmup_frames)])
    if args.tensorboard:
        elastic_args.append("--tensorboard")
    if args.checkpoint_each_epoch:
        elastic_args.append("--checkpoint-each-epoch")
    if args.rest_zero_rehearsal_probability > 0:
        elastic_args.extend(
            ["--rest-zero-rehearsal-probability", str(args.rest_zero_rehearsal_probability)]
        )
    if args.sequence_name:
        for name in args.sequence_name:
            elastic_args.extend(["--sequence-name", name])
    run_step(elastic_args)
    checkpoint = train_pnet_iteration(
        config,
        args.results_dir,
        args.pnet_iteration,
        args.pnet_epochs,
        args.pnet_time_batch_size,
        args.pnet_active_weight,
        args.pnet_active_threshold,
        args.pnet_inactive_weight,
        args.feature_cache_dir,
        args.topology_cache,
        args.sequence_name,
    )
    predict_pnet_iteration(
        config,
        args.results_dir,
        args.pnet_iteration,
        checkpoint,
        args.pnet_alpha_cutoff,
        args.feature_cache_dir,
        args.topology_cache,
        args.sequence_name,
    )


def run_normal(args: argparse.Namespace, config: Path) -> None:
    train_config = MainConfig(str(config))
    target_dir = None
    final_pnet_dir = None

    for iteration in range(train_config.training_iteration):
        train_elastic_iteration(
            config,
            args.gpu_id,
            iteration,
            target_dir,
            args.sequence_name,
            args.results_dir,
            args.elasticity_epochs,
            args.elasticity_initial_epochs,
            args.elasticity_later_epochs,
            args.target_input_noise_std,
            args.target_noise_as_target,
            args.target_noise_smoothing_steps,
            args.warmup_frames,
            args.tensorboard,
            args.checkpoint_each_epoch,
            args.rest_zero_rehearsal_probability,
        )
        checkpoint = train_pnet_iteration(
            config,
            args.results_dir,
            iteration,
            args.pnet_epochs,
            args.pnet_time_batch_size,
            args.pnet_active_weight,
            args.pnet_active_threshold,
            args.pnet_inactive_weight,
            args.feature_cache_dir,
            args.topology_cache,
            args.sequence_name,
        )
        target_dir = predict_pnet_iteration(
            config,
            args.results_dir,
            iteration,
            checkpoint,
            args.pnet_alpha_cutoff,
            args.feature_cache_dir,
            args.topology_cache,
            args.sequence_name,
        )
        final_pnet_dir = target_dir

    if not args.skip_final_predict and final_pnet_dir is not None:
        predict_args = [
            sys.executable,
            "predict_epnet.py",
            "--config",
            str(config),
            "--gpu_id",
            str(args.gpu_id),
            "--plasticity-dir",
            str(final_pnet_dir),
            "--out-dir",
            str(args.results_dir / "prediction_epnet"),
            "--results-dir",
            str(args.results_dir),
        ]
        if args.warmup_frames > 0:
            predict_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.sequence_name:
            for name in args.sequence_name:
                predict_args.extend(["--sequence-name", name])
        run_step(predict_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EPNet training workflows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["normal", "debug"], default="normal")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--elasticity-epochs", type=int, default=None)
    parser.add_argument("--elasticity-initial-epochs", type=int, default=None)
    parser.add_argument("--elasticity-later-epochs", type=int, default=None)
    parser.add_argument("--pnet-epochs", type=int, default=50)
    parser.add_argument("--pnet-time-batch-size", type=int, default=1)
    parser.add_argument("--pnet-iteration", type=int, default=0)
    parser.add_argument("--pnet-active-weight", type=float, default=1.0)
    parser.add_argument("--pnet-active-threshold", type=float, default=1e-4)
    parser.add_argument("--pnet-inactive-weight", type=float, default=1e-3)
    parser.add_argument("--pnet-alpha-cutoff", type=float, default=0.05)
    parser.add_argument("--feature-cache-dir", default=r"results\feature_cache\pnet")
    parser.add_argument("--topology-cache", default=r"results\cache\topology.npz")
    parser.add_argument("--results-dir", default=r"results")
    parser.add_argument("--target-input-noise-std", type=float, default=0.0)
    parser.add_argument("--target-noise-as-target", action="store_true")
    parser.add_argument("--target-noise-smoothing-steps", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--checkpoint-each-epoch", action="store_true")
    parser.add_argument("--skip-final-predict", action="store_true")
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--rest-zero-rehearsal-probability", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = project_path(args.config)
    args.results_dir = project_path(args.results_dir)
    args.feature_cache_dir = project_path(args.feature_cache_dir) if args.feature_cache_dir else None
    args.topology_cache = project_path(args.topology_cache) if args.topology_cache else None

    print(f"[pipeline] mode={args.mode} pnet_backend=tensorflow")
    if args.mode == "debug":
        run_debug(args, config)
    else:
        run_normal(args, config)


if __name__ == "__main__":
    main()
