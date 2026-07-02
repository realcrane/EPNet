from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from global_vars import ROOT_DIR
from model.build import build_ncs_model
from predict_pnet_tf import load_model, predict_threshold
from epnet.data import ElasticityDataset
from epnet.tf_features import load_or_prepare_features_np, load_sorted_npy
from epnet.topology import get_topology
from train_pnet_tf import (
    load_local_body_features_np,
    load_motion_features_np,
    make_stage_feature,
    material_from_config,
)
from utils.config import MainConfig


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_gpu(gpu_id: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")


def predict_pnet_stage(
    *,
    model,
    config: MainConfig,
    topology,
    material: dict,
    garment_dir: Path,
    input_rest_dir: Path | None,
    motion_dir: Path | None,
    out_dir: Path,
    alpha_dir: Path,
    feature_cache_dir: Path | None,
    body_model,
    stage: int,
    alpha_cutoff: float,
    node_position_mode: str,
    output_mode: str,
    motion_feature_mode: str,
    local_body_feature_mode: str,
    stage_feature_mode: str,
    stage_normalizer: float,
    sequence_names: list[str] | None,
) -> None:
    garment_data = load_sorted_npy(garment_dir)
    input_rest_data = load_sorted_npy(input_rest_dir) if input_rest_dir else {}
    items = list(garment_data.items())
    if sequence_names:
        selected = {Path(name).stem for name in sequence_names}
        items = [(name, value) for name, value in items if Path(name).stem in selected]

    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    stage_feature = make_stage_feature(float(stage), stage_normalizer, stage_feature_mode)

    for name, positions in items:
        cache_name = f"{name}_process_stage{stage}"
        features = load_or_prepare_features_np(
            cache_name,
            positions,
            topology,
            material=material,
            cache_dir=feature_cache_dir,
        )
        motion_features = load_motion_features_np(
            motion_dir,
            name,
            features["theta_seq"].shape[0],
            config.body.input_joints,
            motion_feature_mode,
        )
        local_body_features = load_local_body_features_np(
            body_model,
            motion_dir,
            name,
            features,
            features["theta_seq"].shape[0],
            local_body_feature_mode,
            cache_dir=feature_cache_dir,
        )
        threshold, alpha = predict_threshold(
            model,
            features,
            alpha_cutoff=alpha_cutoff,
            node_position_mode=node_position_mode,
            output_mode=output_mode,
            motion_features=motion_features,
            local_body_features=local_body_features,
            stage_feature=stage_feature,
            input_rest=input_rest_data.get(name),
        )
        np.save(out_dir / f"{name}.npy", threshold)
        np.save(alpha_dir / f"{name}.npy", alpha)
        print(f"[process] saved P-Net stage {stage} rest:", out_dir / f"{name}.npy")


def predict_elastic_stage(
    *,
    config: MainConfig,
    topology,
    rest_dir: Path,
    out_dir: Path,
    checkpoint_results_dir: Path,
    stage: int,
    sequence_names: list[str] | None,
    warmup_frames: int,
) -> None:
    edge_count = topology.edge_count
    model = build_ncs_model(config, edge_count)
    data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
    )
    data.load_target(rest_dir)

    checkpoint_path = (
        checkpoint_results_dir
        / "checkpoints"
        / config.experiment.elasticity_checkpoint
        / f"{config.name}_{stage}"
    )
    print("[process] loading E-Net checkpoint:", checkpoint_path)
    status = model.load_weights(str(checkpoint_path))
    if hasattr(status, "expect_partial"):
        status.expect_partial()

    garment_dir = out_dir / "elasticity" / f"iteration_{stage}" / "garment"
    garment_dir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(data):
        _, garment, _, _, frames = model.predict(batch, w=1.0)
        name = Path(data.files_name[i]).stem
        frame_count = int(frames[0])
        export_count = frame_count + int(data.warmup_frames)
        garment_np = np.asarray(garment[0])[-export_count:]
        np.save(garment_dir / f"{name}.npy", garment_np)
        print(f"[process] saved E-Net stage {stage} garment:", garment_dir / f"{name}.npy")


def main() -> None:
    args = parse_args()
    config = MainConfig(str(project_path(args.config)))
    topology = get_topology(config, cache_path=project_path(args.topology_cache) if args.topology_cache else None)
    material = material_from_config(config)
    pnet, meta = load_model(project_path(args.pnet_ckpt))

    motion_dir = project_path(args.motion_dir) if args.motion_dir else None
    out_dir = project_path(args.out_dir)
    feature_cache_dir = project_path(args.feature_cache_dir) if args.feature_cache_dir else None
    body_model = None
    local_body_feature_mode = args.local_body_feature_mode or meta.get("local_body_feature_mode", "none")
    if local_body_feature_mode != "none":
        body_model = build_ncs_model(config, topology.edge_count)

    current_garment_dir = project_path(args.initial_garment_dir)
    prev_rest_dir = project_path(args.initial_rest_dir) if args.initial_rest_dir else None
    for stage in range(1, args.stages + 1):
        rest_dir = out_dir / "plasticity" / f"iteration_{stage}"
        alpha_dir = out_dir / "alpha" / f"iteration_{stage}"
        predict_pnet_stage(
            model=pnet,
            config=config,
            topology=topology,
            material=material,
            garment_dir=current_garment_dir,
            input_rest_dir=prev_rest_dir,
            motion_dir=motion_dir,
            out_dir=rest_dir,
            alpha_dir=alpha_dir,
            feature_cache_dir=feature_cache_dir,
            body_model=body_model,
            stage=stage,
            alpha_cutoff=args.alpha_cutoff,
            node_position_mode=args.node_position_mode or meta.get("node_position_mode", "absolute"),
            output_mode=args.output_mode or meta.get("output_mode", "direct"),
            motion_feature_mode=args.motion_feature_mode or meta.get("motion_feature_mode", "none"),
            local_body_feature_mode=local_body_feature_mode,
            stage_feature_mode=args.stage_feature_mode or meta.get("stage_feature_mode", "none"),
            stage_normalizer=args.stage_normalizer,
            sequence_names=args.sequence_name,
        )
        predict_elastic_stage(
            config=config,
            topology=topology,
            rest_dir=rest_dir,
            out_dir=out_dir,
            checkpoint_results_dir=project_path(args.enet_results_dir),
            stage=stage,
            sequence_names=args.sequence_name,
            warmup_frames=args.warmup_frames,
        )
        prev_rest_dir = rest_dir
        current_garment_dir = out_dir / "elasticity" / f"iteration_{stage}" / "garment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative EPNet P-Net/E-Net process inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--pnet-ckpt", required=True)
    parser.add_argument("--enet-results-dir", required=True)
    parser.add_argument("--initial-garment-dir", required=True)
    parser.add_argument("--initial-rest-dir", default=None)
    parser.add_argument("--motion-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--alpha-cutoff", type=float, default=0.0)
    parser.add_argument("--node-position-mode", choices=("absolute", "displacement", "none"), default=None)
    parser.add_argument("--output-mode", choices=("alpha", "direct"), default=None)
    parser.add_argument("--motion-feature-mode", choices=("none", "summary"), default=None)
    parser.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default=None)
    parser.add_argument("--stage-feature-mode", choices=("none", "scalar"), default=None)
    parser.add_argument("--stage-normalizer", type=float, default=3.0)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--topology-cache", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_gpu(args.gpu_id)
    main()
