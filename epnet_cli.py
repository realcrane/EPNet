from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def configure_gpu(gpu_id: str | None) -> None:
    if gpu_id is None:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def resolve_project_path(path: str | Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def passthrough_args(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def existing_outputs_complete(directory: Path, sequence_names: list[str] | None) -> bool:
    if not directory.is_dir():
        return False
    if sequence_names:
        return all((directory / f"{Path(name).stem}.npy").is_file() for name in sequence_names)
    return any(directory.glob("*.npy"))


def run_project_script(script_name: str, script_args: list[str], gpu_id: str | None = None) -> None:
    configure_gpu(gpu_id)
    sys.argv = [str(ROOT / script_name), *script_args]
    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        runpy.run_path(str(ROOT / script_name), run_name="__main__")
    finally:
        os.chdir(old_cwd)


def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--plasticity-dir", default=r"results\rest")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--elasticity-iteration", type=int, default=None)
    parser.add_argument("--collision-projection-threshold", type=float, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EPNet training and prediction entry point.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Run the EPNet training workflow.")
    train.add_argument("--mode", choices=["normal", "debug"], default="normal")
    train.add_argument("--config", required=True)
    train.add_argument("--gpu-id", default="0")
    train.add_argument("--elasticity-epochs", type=int, default=None)
    train.add_argument("--elasticity-initial-epochs", type=int, default=None)
    train.add_argument("--elasticity-later-epochs", type=int, default=None)
    train.add_argument("--pnet-epochs", type=int, default=50)
    train.add_argument("--pnet-time-batch-size", type=int, default=1)
    train.add_argument("--pnet-iteration", type=int, default=0)
    train.add_argument("--pnet-active-weight", type=float, default=1.0)
    train.add_argument("--pnet-active-threshold", type=float, default=1e-4)
    train.add_argument("--pnet-inactive-weight", type=float, default=1e-3)
    train.add_argument("--pnet-alpha-cutoff", type=float, default=0.05)
    train.add_argument("--feature-cache-dir", default=r"results\feature_cache\pnet")
    train.add_argument("--topology-cache", default=r"results\cache\topology.npz")
    train.add_argument("--results-dir", default=r"results")
    train.add_argument("--target-input-noise-std", type=float, default=0.0)
    train.add_argument("--target-noise-as-target", action="store_true")
    train.add_argument("--target-noise-smoothing-steps", type=int, default=0)
    train.add_argument("--warmup-frames", type=int, default=0)
    train.add_argument("--tensorboard", action="store_true")
    train.add_argument("--checkpoint-each-epoch", action="store_true")
    train.add_argument("--skip-final-predict", action="store_true")
    train.add_argument("--sequence-name", action="append", default=None)
    train.add_argument("--rest-zero-rehearsal-probability", type=float, default=0.0)

    train_elastic = sub.add_parser("train-elastic", help="Train the elastic NCS model.")
    train_elastic.add_argument("--config", required=True)
    train_elastic.add_argument("--gpu-id", default="0")
    train_elastic.add_argument("--start-iteration", type=int, default=None)
    train_elastic.add_argument("--end-iteration", type=int, default=None)
    train_elastic.add_argument("--target-dir", default=None)
    train_elastic.add_argument("--sequence-name", action="append", default=None)
    train_elastic.add_argument("--results-dir", default=None)
    train_elastic.add_argument("--elasticity-epochs", type=int, default=None)
    train_elastic.add_argument("--elasticity-initial-epochs", type=int, default=None)
    train_elastic.add_argument("--elasticity-later-epochs", type=int, default=None)
    train_elastic.add_argument("--target-input-noise-std", type=float, default=0.0)
    train_elastic.add_argument("--target-noise-as-target", action="store_true")
    train_elastic.add_argument("--target-noise-smoothing-steps", type=int, default=0)
    train_elastic.add_argument("--warmup-frames", type=int, default=0)
    train_elastic.add_argument("--tensorboard", action="store_true")
    train_elastic.add_argument("--checkpoint-each-epoch", action="store_true")
    train_elastic.add_argument("--preview-epoch", type=int, action="append", default=None)
    train_elastic.add_argument("--rest-zero-rehearsal-probability", type=float, default=0.0)

    export_elastic = sub.add_parser("export-elastic", help="Export elastic checkpoints without training.")
    export_elastic.add_argument("--config", required=True)
    export_elastic.add_argument("--gpu-id", default="0")
    export_elastic.add_argument("--checkpoint-results-dir", required=True)
    export_elastic.add_argument("--out-dir", required=True)
    export_elastic.add_argument("--start-iteration", type=int, default=0)
    export_elastic.add_argument("--end-iteration", type=int, default=None)
    export_elastic.add_argument("--sequence-name", action="append", default=None)
    export_elastic.add_argument("--warmup-frames", type=int, default=0)
    export_elastic.add_argument("--force-zero-rest", action="store_true")

    export_lbs = sub.add_parser("export-lbs", help="Export LBS garment motion for P-Net input experiments.")
    export_lbs.add_argument("--config", required=True)
    export_lbs.add_argument("--gpu-id", default="0")
    export_lbs.add_argument("--out-dir", required=True)
    export_lbs.add_argument("--checkpoint-results-dir", default=None)
    export_lbs.add_argument("--elasticity-iteration", type=int, default=0)
    export_lbs.add_argument("--sequence-name", action="append", default=None)
    export_lbs.add_argument("--warmup-frames", type=int, default=0)

    train_pnet = sub.add_parser("train-pnet", help="Train P-Net from EPNet garment/threshold .npy files.")
    train_pnet.add_argument("args", nargs=argparse.REMAINDER)

    predict_pnet = sub.add_parser("predict-pnet", help="Predict EPNet plasticity .npy files with P-Net.")
    predict_pnet.add_argument("args", nargs=argparse.REMAINDER)

    predict = sub.add_parser("predict", help="Run EPNet prediction using P-Net plasticity files.")
    add_predict_args(predict)

    predict_process = sub.add_parser("predict-process", help="Run iterative P-Net/E-Net process prediction.")
    predict_process.add_argument("args", nargs=argparse.REMAINDER)

    predict_coupled = sub.add_parser("predict-coupled", help="Run in-memory coupled P-Net/E-Net prediction.")
    predict_coupled.add_argument("args", nargs=argparse.REMAINDER)

    predict_proxy = sub.add_parser("predict-proxy", help="Run proxy-garment sparse P-Net prediction.")
    predict_proxy.add_argument("--config", required=True)
    predict_proxy.add_argument("--gpu-id", default="0")
    predict_proxy.add_argument("--checkpoint-results-dir", required=True)
    predict_proxy.add_argument("--pnet-ckpt", required=True)
    predict_proxy.add_argument("--out-dir", required=True)
    predict_proxy.add_argument("--proxy-iteration", type=int, default=3)
    predict_proxy.add_argument("--elasticity-iteration", type=int, default=None)
    predict_proxy.add_argument("--alpha-cutoff", type=float, default=0.05)
    predict_proxy.add_argument("--drive-threshold", type=float, default=None)
    predict_proxy.add_argument("--drive-window", type=int, default=1)
    predict_proxy.add_argument("--drive-min-frames", type=int, default=1)
    predict_proxy.add_argument("--rest-scale", type=float, default=1.0)
    predict_proxy.add_argument("--warmup-frames", type=int, default=0)
    predict_proxy.add_argument("--collision-projection-threshold", type=float, default=None)
    predict_proxy.add_argument("--skip-existing-proxy", action="store_true")
    predict_proxy.add_argument("--skip-existing-pnet", action="store_true")
    predict_proxy.add_argument("--sequence-name", action="append", default=None)
    predict_proxy.add_argument("--topology-cache", default=None)
    predict_proxy.add_argument("--feature-cache-dir", default=None)
    predict_proxy.add_argument("--motion-dir", default=None)
    predict_proxy.add_argument("--motion-feature-mode", choices=("none", "summary"), default=None)
    predict_proxy.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default=None)
    predict_proxy.add_argument("--stage-feature-mode", choices=("none", "scalar"), default=None)
    predict_proxy.add_argument("--stage-normalizer", type=float, default=None)

    train_proxy_pnet = sub.add_parser("train-proxy-pnet", help="Train sparse P-Net from zero-rest proxy garments.")
    train_proxy_pnet.add_argument("--config", required=True)
    train_proxy_pnet.add_argument("--gpu-id", default="0")
    train_proxy_pnet.add_argument("--checkpoint-results-dir", required=True)
    train_proxy_pnet.add_argument("--out-dir", required=True)
    train_proxy_pnet.add_argument("--proxy-iteration", type=int, default=3)
    train_proxy_pnet.add_argument("--target-dir", default=None)
    train_proxy_pnet.add_argument("--epochs", type=int, default=50)
    train_proxy_pnet.add_argument("--lr", type=float, default=1e-4)
    train_proxy_pnet.add_argument("--noise-std", type=float, default=0.05)
    train_proxy_pnet.add_argument("--active-weight", type=float, default=1.0)
    train_proxy_pnet.add_argument("--active-threshold", type=float, default=1e-4)
    train_proxy_pnet.add_argument("--drive-threshold", type=float, default=0.0)
    train_proxy_pnet.add_argument("--inactive-weight", type=float, default=1e-3)
    train_proxy_pnet.add_argument("--time-batch-size", type=int, default=1)
    train_proxy_pnet.add_argument("--output-mode", choices=("alpha", "direct", "mask_direct", "delta_direct"), default="alpha")
    train_proxy_pnet.add_argument("--motion-dir", default=None)
    train_proxy_pnet.add_argument("--motion-feature-mode", choices=("none", "summary"), default="none")
    train_proxy_pnet.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default="none")
    train_proxy_pnet.add_argument("--stage-feature-mode", choices=("none", "scalar"), default="none")
    train_proxy_pnet.add_argument("--stage-normalizer", type=float, default=3.0)
    train_proxy_pnet.add_argument("--validation-sequence-name", action="append", default=None)
    train_proxy_pnet.add_argument("--sequence-name", action="append", default=None)
    train_proxy_pnet.add_argument("--max-sequences", type=int, default=None)
    train_proxy_pnet.add_argument("--topology-cache", default=None)
    train_proxy_pnet.add_argument("--feature-cache-dir", default=None)
    train_proxy_pnet.add_argument("--early-stopping-patience", type=int, default=8)
    train_proxy_pnet.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    train_proxy_pnet.add_argument("--load", default=None)

    sub.add_parser("check-env", help="Check Python dependencies for EPNet.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        script_args = [
            "--mode",
            args.mode,
            "--config",
            resolve_project_path(args.config),
            "--gpu-id",
            args.gpu_id,
            "--pnet-epochs",
            str(args.pnet_epochs),
            "--pnet-time-batch-size",
            str(args.pnet_time_batch_size),
            "--pnet-iteration",
            str(args.pnet_iteration),
            "--pnet-active-weight",
            str(args.pnet_active_weight),
            "--pnet-active-threshold",
            str(args.pnet_active_threshold),
            "--pnet-inactive-weight",
            str(args.pnet_inactive_weight),
            "--pnet-alpha-cutoff",
            str(args.pnet_alpha_cutoff),
            "--results-dir",
            resolve_project_path(args.results_dir),
        ]
        if args.elasticity_epochs is not None:
            script_args.extend(["--elasticity-epochs", str(args.elasticity_epochs)])
        if args.elasticity_initial_epochs is not None:
            script_args.extend(["--elasticity-initial-epochs", str(args.elasticity_initial_epochs)])
        if args.elasticity_later_epochs is not None:
            script_args.extend(["--elasticity-later-epochs", str(args.elasticity_later_epochs)])
        if args.target_input_noise_std > 0:
            script_args.extend(["--target-input-noise-std", str(args.target_input_noise_std)])
        if args.target_noise_as_target:
            script_args.append("--target-noise-as-target")
        if args.target_noise_smoothing_steps > 0:
            script_args.extend(["--target-noise-smoothing-steps", str(args.target_noise_smoothing_steps)])
        if args.warmup_frames > 0:
            script_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.tensorboard:
            script_args.append("--tensorboard")
        if args.checkpoint_each_epoch:
            script_args.append("--checkpoint-each-epoch")
        if args.rest_zero_rehearsal_probability > 0:
            script_args.extend(
                [
                    "--rest-zero-rehearsal-probability",
                    str(args.rest_zero_rehearsal_probability),
                ]
            )
        if args.feature_cache_dir:
            script_args.extend(["--feature-cache-dir", resolve_project_path(args.feature_cache_dir)])
        if args.topology_cache:
            script_args.extend(["--topology-cache", resolve_project_path(args.topology_cache)])
        if args.skip_final_predict:
            script_args.append("--skip-final-predict")
        if args.sequence_name:
            for name in args.sequence_name:
                script_args.extend(["--sequence-name", name])
        run_project_script("train_pipeline.py", script_args, args.gpu_id)
    elif args.cmd == "train-elastic":
        script_args = ["--config", resolve_project_path(args.config), "--gpu_id", args.gpu_id]
        if args.start_iteration is not None:
            script_args.extend(["--start-iteration", str(args.start_iteration)])
        if args.end_iteration is not None:
            script_args.extend(["--end-iteration", str(args.end_iteration)])
        if args.target_dir:
            script_args.extend(["--target-dir", resolve_project_path(args.target_dir)])
        if args.results_dir:
            script_args.extend(["--results-dir", resolve_project_path(args.results_dir)])
        if args.elasticity_epochs is not None:
            script_args.extend(["--elasticity-epochs", str(args.elasticity_epochs)])
        if args.elasticity_initial_epochs is not None:
            script_args.extend(["--elasticity-initial-epochs", str(args.elasticity_initial_epochs)])
        if args.elasticity_later_epochs is not None:
            script_args.extend(["--elasticity-later-epochs", str(args.elasticity_later_epochs)])
        if args.target_input_noise_std > 0:
            script_args.extend(["--target-input-noise-std", str(args.target_input_noise_std)])
        if args.target_noise_as_target:
            script_args.append("--target-noise-as-target")
        if args.target_noise_smoothing_steps > 0:
            script_args.extend(["--target-noise-smoothing-steps", str(args.target_noise_smoothing_steps)])
        if args.warmup_frames > 0:
            script_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.tensorboard:
            script_args.append("--tensorboard")
        if args.checkpoint_each_epoch:
            script_args.append("--checkpoint-each-epoch")
        if args.preview_epoch:
            for epoch in args.preview_epoch:
                script_args.extend(["--preview-epoch", str(epoch)])
        if args.rest_zero_rehearsal_probability > 0:
            script_args.extend(
                [
                    "--rest-zero-rehearsal-probability",
                    str(args.rest_zero_rehearsal_probability),
                ]
            )
        if args.sequence_name:
            for name in args.sequence_name:
                script_args.extend(["--sequence-name", name])
        run_project_script(
            "main.py",
            script_args,
            args.gpu_id,
        )
    elif args.cmd == "export-elastic":
        script_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--checkpoint-results-dir",
            resolve_project_path(args.checkpoint_results_dir),
            "--out-dir",
            resolve_project_path(args.out_dir),
            "--start-iteration",
            str(args.start_iteration),
        ]
        if args.end_iteration is not None:
            script_args.extend(["--end-iteration", str(args.end_iteration)])
        if args.warmup_frames > 0:
            script_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.sequence_name:
            for name in args.sequence_name:
                script_args.extend(["--sequence-name", name])
        if args.force_zero_rest:
            script_args.append("--force-zero-rest")
        run_project_script("export_elastic.py", script_args, args.gpu_id)
    elif args.cmd == "export-lbs":
        script_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--out-dir",
            resolve_project_path(args.out_dir),
            "--elasticity-iteration",
            str(args.elasticity_iteration),
        ]
        if args.checkpoint_results_dir:
            script_args.extend(
                ["--checkpoint-results-dir", resolve_project_path(args.checkpoint_results_dir)]
            )
        if args.warmup_frames > 0:
            script_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.sequence_name:
            for name in args.sequence_name:
                script_args.extend(["--sequence-name", name])
        run_project_script("export_lbs.py", script_args, args.gpu_id)
    elif args.cmd == "train-pnet":
        run_project_script("train_pnet_tf.py", passthrough_args(args.args))
    elif args.cmd == "predict-pnet":
        run_project_script("predict_pnet_tf.py", passthrough_args(args.args))
    elif args.cmd == "predict":
        script_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--plasticity-dir",
            resolve_project_path(args.plasticity_dir),
        ]
        if args.out_dir:
            script_args.extend(["--out-dir", resolve_project_path(args.out_dir)])
        if args.max_sequences is not None:
            script_args.extend(["--max-sequences", str(args.max_sequences)])
        if hasattr(args, "results_dir") and args.results_dir:
            script_args.extend(["--results-dir", resolve_project_path(args.results_dir)])
        if args.warmup_frames > 0:
            script_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.elasticity_iteration is not None:
            script_args.extend(["--elasticity-iteration", str(args.elasticity_iteration)])
        if args.collision_projection_threshold is not None:
            script_args.extend(
                [
                    "--collision-projection-threshold",
                    str(args.collision_projection_threshold),
                ]
            )
        if args.sequence_name:
            for name in args.sequence_name:
                script_args.extend(["--sequence-name", name])
        run_project_script("predict_epnet.py", script_args, args.gpu_id)
    elif args.cmd == "predict-process":
        run_project_script("predict_epnet_process.py", passthrough_args(args.args))
    elif args.cmd == "predict-coupled":
        coupled_args = passthrough_args(args.args)
        gpu_id = None
        for i, value in enumerate(coupled_args[:-1]):
            if value in ("--gpu_id", "--gpu-id"):
                gpu_id = coupled_args[i + 1]
                break
        run_project_script("predict_epnet_coupled.py", coupled_args, gpu_id)
    elif args.cmd == "predict-proxy":
        proxy_iteration = int(args.proxy_iteration)
        final_iteration = (
            proxy_iteration if args.elasticity_iteration is None else int(args.elasticity_iteration)
        )
        out_dir = Path(resolve_project_path(args.out_dir))
        proxy_root = out_dir / "proxy"
        rest_dir = out_dir / "rest"
        alpha_dir = out_dir / "alpha"
        render_dir = out_dir / "render"

        export_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--checkpoint-results-dir",
            resolve_project_path(args.checkpoint_results_dir),
            "--out-dir",
            str(proxy_root),
            "--start-iteration",
            str(proxy_iteration),
            "--end-iteration",
            str(proxy_iteration + 1),
            "--force-zero-rest",
        ]
        if args.sequence_name:
            for name in args.sequence_name:
                export_args.extend(["--sequence-name", name])
        proxy_garment_dir = proxy_root / "elasticity" / f"iteration_{proxy_iteration}" / "garment"
        if args.skip_existing_proxy and existing_outputs_complete(proxy_garment_dir, args.sequence_name):
            print(f"Skip existing proxy garments: {proxy_garment_dir}")
        else:
            run_project_script("export_elastic.py", export_args, args.gpu_id)

        pnet_args = [
            "--config",
            resolve_project_path(args.config),
            "--ckpt",
            resolve_project_path(args.pnet_ckpt),
            "--garment-dir",
            str(proxy_garment_dir),
            "--out-dir",
            str(rest_dir),
            "--alpha-dir",
            str(alpha_dir),
            "--alpha-cutoff",
            str(args.alpha_cutoff),
            "--rest-scale",
            str(args.rest_scale),
        ]
        if args.topology_cache:
            pnet_args.extend(["--topology-cache", resolve_project_path(args.topology_cache)])
        if args.feature_cache_dir:
            pnet_args.extend(["--feature-cache-dir", resolve_project_path(args.feature_cache_dir)])
        if args.motion_dir:
            pnet_args.extend(["--motion-dir", resolve_project_path(args.motion_dir)])
        if args.motion_feature_mode:
            pnet_args.extend(["--motion-feature-mode", args.motion_feature_mode])
        if args.local_body_feature_mode:
            pnet_args.extend(["--local-body-feature-mode", args.local_body_feature_mode])
        if args.stage_feature_mode:
            pnet_args.extend(["--stage-feature-mode", args.stage_feature_mode])
        if args.stage_normalizer is not None:
            pnet_args.extend(["--stage-normalizer", str(args.stage_normalizer)])
        if args.drive_threshold is not None:
            pnet_args.extend(["--drive-threshold", str(args.drive_threshold)])
        pnet_args.extend(["--drive-window", str(args.drive_window)])
        pnet_args.extend(["--drive-min-frames", str(args.drive_min_frames)])
        pnet_args.extend(["--stage-value", str(proxy_iteration)])
        if args.sequence_name:
            for name in args.sequence_name:
                pnet_args.extend(["--sequence-name", name])
        if args.skip_existing_pnet and existing_outputs_complete(rest_dir, args.sequence_name):
            print(f"Skip existing P-Net plasticity: {rest_dir}")
        else:
            run_project_script("predict_pnet_tf.py", pnet_args, args.gpu_id)

        predict_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--plasticity-dir",
            str(rest_dir),
            "--out-dir",
            str(render_dir),
            "--results-dir",
            resolve_project_path(args.checkpoint_results_dir),
            "--elasticity-iteration",
            str(final_iteration),
        ]
        if args.warmup_frames > 0:
            predict_args.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.collision_projection_threshold is not None:
            predict_args.extend(
                [
                    "--collision-projection-threshold",
                    str(args.collision_projection_threshold),
                ]
            )
        if args.sequence_name:
            for name in args.sequence_name:
                predict_args.extend(["--sequence-name", name])
        run_project_script("predict_epnet.py", predict_args, args.gpu_id)
    elif args.cmd == "train-proxy-pnet":
        proxy_iteration = int(args.proxy_iteration)
        out_dir = Path(resolve_project_path(args.out_dir))
        proxy_root = out_dir / "proxy"
        target_dir = (
            Path(resolve_project_path(args.target_dir))
            if args.target_dir
            else Path(resolve_project_path(args.checkpoint_results_dir))
            / "elasticity"
            / f"iteration_{proxy_iteration}"
            / "threshold"
        )
        ckpt_dir = out_dir / "checkpoints" / "pnet" / f"proxy_iteration_{proxy_iteration}"
        last_ckpt = ckpt_dir / "pnet.weights.h5"
        best_ckpt = ckpt_dir.with_name(f"{ckpt_dir.name}_best") / "pnet.weights.h5"

        export_args = [
            "--config",
            resolve_project_path(args.config),
            "--gpu_id",
            args.gpu_id,
            "--checkpoint-results-dir",
            resolve_project_path(args.checkpoint_results_dir),
            "--out-dir",
            str(proxy_root),
            "--start-iteration",
            str(proxy_iteration),
            "--end-iteration",
            str(proxy_iteration + 1),
            "--force-zero-rest",
        ]
        if args.sequence_name:
            for name in args.sequence_name:
                export_args.extend(["--sequence-name", name])
        run_project_script("export_elastic.py", export_args, args.gpu_id)

        train_args = [
            "--config",
            resolve_project_path(args.config),
            "--garment-dir",
            str(proxy_root / "elasticity" / f"iteration_{proxy_iteration}" / "garment"),
            "--target-dir",
            str(target_dir),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--noise-std",
            str(args.noise_std),
            "--active-weight",
            str(args.active_weight),
            "--active-threshold",
            str(args.active_threshold),
            "--drive-threshold",
            str(args.drive_threshold),
            "--inactive-weight",
            str(args.inactive_weight),
            "--time-batch-size",
            str(args.time_batch_size),
            "--node-position-mode",
            "absolute",
            "--output-mode",
            args.output_mode,
            "--motion-feature-mode",
            args.motion_feature_mode,
            "--local-body-feature-mode",
            args.local_body_feature_mode,
            "--stage-feature-mode",
            args.stage_feature_mode,
            "--stage-normalizer",
            str(args.stage_normalizer),
            "--stage-value",
            str(proxy_iteration),
            "--out",
            str(last_ckpt),
            "--best-out",
            str(best_ckpt),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
        ]
        if args.load:
            train_args.extend(["--load", resolve_project_path(args.load)])
        if args.motion_dir:
            train_args.extend(["--motion-dir", resolve_project_path(args.motion_dir)])
        if args.topology_cache:
            train_args.extend(["--topology-cache", resolve_project_path(args.topology_cache)])
        if args.feature_cache_dir:
            train_args.extend(["--feature-cache-dir", resolve_project_path(args.feature_cache_dir)])
        if args.validation_sequence_name:
            for name in args.validation_sequence_name:
                train_args.extend(["--validation-sequence-name", name])
        if args.sequence_name:
            for name in args.sequence_name:
                train_args.extend(["--sequence-name", name])
        if args.max_sequences is not None:
            train_args.extend(["--max-sequences", str(args.max_sequences)])
        run_project_script("train_pnet_tf.py", train_args, args.gpu_id)
    elif args.cmd == "check-env":
        run_project_script("check_epnet_env.py", [])
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
