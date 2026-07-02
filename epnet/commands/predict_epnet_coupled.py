from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from epnet.global_vars import CHECKPOINTS_DIR, ROOT_DIR
from model.build import build_ncs_model
from predict_pnet_tf import load_model as load_pnet_model
from predict_pnet_tf import predict_threshold
from epnet.data.elasticity import ElasticityDataset
from epnet.tf_features import load_or_prepare_features_np
from epnet.topology import get_topology
from epnet.commands.train_pnet_tf import (
    load_local_body_features_np,
    load_motion_features_np,
    make_stage_feature,
    material_from_config,
)
from utils import debug
from utils.IO import writePC2Frames
from utils.collision_projection import project_cloth_outside_body
from utils.config import MainConfig


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_gpu(gpu_id: str | None) -> None:
    if gpu_id is None:
        return
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def overwrite_pc2(path: Path, frames: np.ndarray) -> None:
    if path.exists():
        path.unlink()
    writePC2Frames(str(path), frames)


def fill_plasticity(out: np.ndarray, target: np.ndarray) -> None:
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    pad_len = out.shape[0] - target.shape[0]
    if pad_len < 0:
        target = target[-out.shape[0] :]
        pad_len = 0
    out[:pad_len, :, :] = target[0:1, :, None]
    out[pad_len:, :, :] = target[..., None]


def save_prediction_frames(
    elasticity_model,
    config: MainConfig,
    batch,
    out_root: Path,
    name: str,
    collision_projection_threshold: float | None,
) -> None:
    body, garment, unskinned, _, frames = elasticity_model.predict(batch, w=1.0)
    frame_count = int(frames[0])
    warmup_frames = batch[1]["target"].shape[0] - frame_count
    export_count = frame_count + max(0, int(warmup_frames))

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
    garment_projected = project_cloth_outside_body(
        body,
        garment,
        elasticity_model.body.faces,
        projection_threshold,
    )

    np.save(render_path / f"{name}_garment.npy", garment_projected)
    np.save(render_path / f"{name}_threshold.npy", threshold)
    overwrite_pc2(render_path / "body.pc2", body)
    overwrite_pc2(render_path / "tshirt.pc2", garment_projected)
    overwrite_pc2(render_path / "tshirt_unskinned.pc2", unskinned)


def main() -> None:
    args = parse_args()
    configure_gpu(args.gpu_id)

    config = MainConfig(str(project_path(args.config)))
    topology = get_topology(config, cache_path=project_path(args.topology_cache) if args.topology_cache else None)
    edge_count = topology.edge_count
    material = material_from_config(config)
    feature_cache_dir = project_path(args.feature_cache_dir) if args.feature_cache_dir else None
    motion_dir = project_path(args.motion_dir) if args.motion_dir else None

    print("Building shared E-Net...")
    elasticity_model = build_ncs_model(config, edge_count)
    checkpoint_root = project_path(args.results_dir) / "checkpoints" if args.results_dir else Path(CHECKPOINTS_DIR)
    checkpoint_path = (
        checkpoint_root
        / config.experiment.elasticity_checkpoint
        / f"{config.name}_{int(args.elasticity_iteration)}"
    )
    print("Loading E-Net checkpoint:", checkpoint_path)
    status = elasticity_model.load_weights(str(checkpoint_path))
    if hasattr(status, "expect_partial"):
        status.expect_partial()

    print("Loading P-Net checkpoint:", project_path(args.pnet_ckpt))
    pnet, pnet_meta = load_pnet_model(project_path(args.pnet_ckpt))
    output_mode = args.output_mode or pnet_meta.get("output_mode", "delta_direct")
    node_position_mode = args.node_position_mode or pnet_meta.get("node_position_mode", "absolute")
    stage_feature_mode = args.stage_feature_mode or pnet_meta.get("stage_feature_mode", "none")
    stage_normalizer = args.stage_normalizer if args.stage_normalizer is not None else pnet_meta.get("stage_normalizer", 3.0)
    stage_value = args.stage_value if args.stage_value is not None else pnet_meta.get("stage_value", args.elasticity_iteration)
    motion_feature_mode = args.motion_feature_mode or pnet_meta.get("motion_feature_mode", "none")
    local_body_feature_mode = args.local_body_feature_mode or pnet_meta.get("local_body_feature_mode", "none")

    body_model = None
    if local_body_feature_mode != "none":
        body_model = elasticity_model

    proxy_data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        max_sequences=args.max_sequences,
        sequence_names=args.sequence_name,
        warmup_frames=0,
    )
    final_data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        max_sequences=args.max_sequences,
        sequence_names=args.sequence_name,
        warmup_frames=args.warmup_frames,
    )

    out_root = project_path(args.out_dir)
    plasticity_dir = out_root / "rest"
    alpha_dir = out_root / "alpha"
    render_root = out_root / "render"
    plasticity_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    print("Start coupled prediction...")
    for idx, proxy_batch in enumerate(proxy_data):
        name = Path(proxy_data.files_name[idx]).stem
        final_batch = final_data[idx]
        print("File name: [", name, "]")

        _, proxy_garment, _, _, proxy_frames = elasticity_model.predict(proxy_batch, w=1.0)
        frame_count = int(proxy_frames[0])
        proxy_positions = np.array(proxy_garment[0])[-frame_count:].astype(np.float32)

        features = load_or_prepare_features_np(
            name,
            proxy_positions,
            topology,
            material=material,
            cache_dir=feature_cache_dir,
            rebuild=args.rebuild_feature_cache,
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
            rebuild=args.rebuild_feature_cache,
        )
        stage_feature = make_stage_feature(stage_value, stage_normalizer, stage_feature_mode)

        threshold, alpha = predict_threshold(
            pnet,
            features,
            absolute=False,
            alpha_cutoff=args.alpha_cutoff,
            node_position_mode=node_position_mode,
            output_mode=output_mode,
            motion_features=motion_features,
            local_body_features=local_body_features,
            stage_feature=stage_feature,
            drive_threshold=args.drive_threshold,
            drive_window=args.drive_window,
            drive_min_frames=args.drive_min_frames,
        )
        if args.rest_scale != 1.0:
            threshold = (threshold * float(args.rest_scale)).astype(np.float32)
        np.save(plasticity_dir / f"{name}.npy", threshold)
        np.save(alpha_dir / f"{name}.npy", alpha)

        fill_plasticity(final_batch[1]["target"], threshold)
        fill_plasticity(final_batch[1]["target_input"], threshold)
        save_prediction_frames(
            elasticity_model,
            config,
            final_batch,
            render_root,
            name,
            args.collision_projection_threshold,
        )
    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coupled in-memory EPNet prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--pnet-ckpt", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--elasticity-iteration", type=int, default=3)
    parser.add_argument("--alpha-cutoff", type=float, default=0.005)
    parser.add_argument("--drive-threshold", type=float, default=0.05)
    parser.add_argument("--drive-window", type=int, default=1)
    parser.add_argument("--drive-min-frames", type=int, default=1)
    parser.add_argument("--rest-scale", type=float, default=1.0)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--collision-projection-threshold", type=float, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--topology-cache", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--motion-dir", default=None)
    parser.add_argument("--motion-feature-mode", choices=("none", "summary"), default=None)
    parser.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default=None)
    parser.add_argument("--stage-feature-mode", choices=("none", "scalar"), default=None)
    parser.add_argument("--stage-normalizer", type=float, default=None)
    parser.add_argument("--stage-value", type=float, default=None)
    parser.add_argument("--node-position-mode", choices=("absolute", "displacement", "none"), default=None)
    parser.add_argument("--output-mode", choices=("alpha", "direct", "mask_direct", "delta_direct"), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
