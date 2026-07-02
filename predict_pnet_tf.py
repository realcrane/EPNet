from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from global_vars import ROOT_DIR
from model.build import build_ncs_model
from epnet.tf_features import load_or_prepare_features_np, load_sorted_npy, wrap_to_pi_np
from epnet.tf_model import make_pnet_model
from epnet.topology import get_topology
from train_pnet_tf import (
    load_local_body_features_np,
    load_motion_features_np,
    make_node_features,
    make_stage_feature,
    material_from_config,
)
from utils.config import MainConfig


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def predict_threshold(
    model,
    features: dict[str, np.ndarray],
    absolute: bool = False,
    alpha_cutoff: float = 0.0,
    node_position_mode: str = "absolute",
    output_mode: str = "alpha",
    motion_features: np.ndarray | None = None,
    local_body_features: np.ndarray | None = None,
    stage_feature: np.ndarray | None = None,
    input_rest: np.ndarray | None = None,
    drive_threshold: float = 0.0,
    drive_window: int = 1,
    drive_min_frames: int = 1,
):
    theta_seq = features["theta_seq"].astype(np.float32)
    edge_index = tf.convert_to_tensor(features["edge_index"], dtype=tf.int32)
    t_count, hinge_count, _ = theta_seq.shape

    theta_rest = np.zeros((t_count, hinge_count, 1), dtype=np.float32)
    if input_rest is not None:
        input_rest = input_rest.astype(np.float32)
        if input_rest.ndim == 2:
            input_rest = input_rest[..., None]
        if input_rest.shape[0] < t_count:
            pad = np.broadcast_to(input_rest[-1:], (t_count - input_rest.shape[0], *input_rest.shape[1:]))
            input_rest = np.concatenate([input_rest, pad], axis=0)
        input_rest = input_rest[:t_count]
    alpha_pred = np.zeros_like(theta_rest)
    drive_window = max(1, int(drive_window))
    drive_min_frames = max(1, int(drive_min_frames))
    drive_history = np.zeros((t_count, hinge_count, 1), dtype=bool)
    if t_count < 2:
        threshold = np.abs(theta_rest) if absolute else theta_rest
        return threshold[..., 0], alpha_pred[..., 0]

    if output_mode in ("direct", "mask_direct", "delta_direct"):
        for t in range(1, t_count - 1):
            if output_mode == "delta_direct":
                rest_prev = theta_rest[t - 1] if input_rest is None else input_rest[t - 1]
                rest_curr = theta_rest[t] if input_rest is None else input_rest[t]
            else:
                rest_prev = np.zeros_like(theta_rest[t - 1]) if input_rest is None else input_rest[t - 1]
                rest_curr = np.zeros_like(theta_rest[t]) if input_rest is None else input_rest[t]
            node_features = make_node_features(
                theta_seq,
                rest_prev,
                rest_curr,
                features,
                t,
                node_position_mode,
                motion_features,
                local_body_features,
                stage_feature,
            )
            edge_features = features["edge_feat_cache"][t].astype(np.float32)
            raw_pred = model(
                (
                    tf.convert_to_tensor(node_features, dtype=tf.float32),
                    edge_index,
                    tf.convert_to_tensor(edge_features, dtype=tf.float32),
                ),
                training=False,
            ).numpy()
            if output_mode == "mask_direct":
                mask = 1.0 / (1.0 + np.exp(-raw_pred[:, :1]))
                rest_hat = raw_pred[:, 1:2]
                if alpha_cutoff > 0.0:
                    rest_hat = np.where(mask >= alpha_cutoff, rest_hat, 0.0)
                alpha_pred[t] = mask
            elif output_mode == "delta_direct":
                drive_now = wrap_to_pi_np(theta_seq[t] - rest_curr)
                drive_history[t] = np.abs(drive_now) > float(drive_threshold)
                window_start = max(1, t - drive_window + 1)
                sustained_mask = np.sum(drive_history[window_start : t + 1], axis=0) >= drive_min_frames
                drive_mask = drive_history[t] & sustained_mask
                delta = np.where(drive_mask, raw_pred, 0.0)
                if alpha_cutoff > 0.0:
                    delta = np.where(np.abs(delta) >= alpha_cutoff, delta, 0.0)
                candidate = theta_rest[t] + delta
                theta_rest[t + 1] = np.sign(candidate) * np.maximum(np.abs(theta_rest[t]), np.abs(candidate))
                alpha_pred[t] = np.abs(delta) > 0.0
                continue
            else:
                rest_hat = raw_pred
                alpha_pred[t] = np.abs(rest_hat) > 0.0
            if output_mode == "direct" and alpha_cutoff > 0.0:
                rest_hat = np.where(np.abs(rest_hat) >= alpha_cutoff, rest_hat, 0.0)
            theta_rest[t] = rest_hat

        threshold = np.abs(theta_rest) if absolute else theta_rest
        return threshold[..., 0].astype(np.float32), alpha_pred[..., 0].astype(np.float32)

    for t in range(1, t_count - 1):
        rest_prev = theta_rest[t - 1] if input_rest is None else input_rest[t - 1]
        rest_curr = theta_rest[t] if input_rest is None else input_rest[t]
        node_features = make_node_features(
            theta_seq,
            rest_prev,
            rest_curr,
            features,
            t,
            node_position_mode,
            motion_features,
            local_body_features,
            stage_feature,
        )
        edge_features = features["edge_feat_cache"][t].astype(np.float32)
        alpha_hat = tf.sigmoid(
            model(
                (
                    tf.convert_to_tensor(node_features, dtype=tf.float32),
                    edge_index,
                    tf.convert_to_tensor(edge_features, dtype=tf.float32),
                ),
                training=False,
            )
        ).numpy()
        if alpha_cutoff > 0.0:
            alpha_hat = np.where(alpha_hat >= alpha_cutoff, alpha_hat, 0.0)
        delta = alpha_hat * np.abs(wrap_to_pi_np(theta_seq[t] - theta_rest[t])) * np.sign(
            theta_seq[t] - theta_rest[t]
        )
        candidate = theta_rest[t] + delta
        theta_rest[t + 1] = np.sign(candidate) * np.maximum(np.abs(theta_rest[t]), np.abs(candidate))
        alpha_pred[t] = alpha_hat

    threshold = np.abs(theta_rest) if absolute else theta_rest
    return threshold[..., 0].astype(np.float32), alpha_pred[..., 0].astype(np.float32)


def load_model(checkpoint_path: Path):
    meta_path = checkpoint_path.with_suffix(".json")
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model, _ = make_pnet_model(meta.get("model_config"))
    model.load_weights(str(checkpoint_path))
    return model, meta


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    checkpoint_path = project_path(args.ckpt)
    garment_dir = project_path(args.garment_dir)
    input_rest_dir = project_path(args.input_rest_dir) if args.input_rest_dir else None
    out_dir = project_path(args.out_dir)
    alpha_dir = project_path(args.alpha_dir) if args.alpha_dir else None
    feature_cache_dir = project_path(args.feature_cache_dir) if args.feature_cache_dir else None
    topology_cache = project_path(args.topology_cache) if args.topology_cache else None

    config = MainConfig(str(config_path))
    topology = get_topology(config, cache_path=topology_cache, rebuild=args.rebuild_topology)
    garment_data = load_sorted_npy(garment_dir)
    input_rest_data = load_sorted_npy(input_rest_dir) if input_rest_dir else {}
    model, meta = load_model(checkpoint_path)
    node_position_mode = args.node_position_mode or meta.get("node_position_mode", "absolute")
    output_mode = args.output_mode or meta.get("output_mode", "alpha")
    motion_feature_mode = args.motion_feature_mode or meta.get("motion_feature_mode", "none")
    local_body_feature_mode = args.local_body_feature_mode or meta.get("local_body_feature_mode", "none")
    stage_feature_mode = args.stage_feature_mode or meta.get("stage_feature_mode", "none")
    stage_normalizer = args.stage_normalizer if args.stage_normalizer is not None else meta.get("stage_normalizer", 3.0)
    stage_value = args.stage_value if args.stage_value is not None else meta.get("stage_value", 1.0)
    drive_threshold = args.drive_threshold if args.drive_threshold is not None else meta.get("drive_threshold", 0.0)
    motion_dir = project_path(args.motion_dir) if args.motion_dir else None
    if motion_dir is None and meta.get("motion_dir"):
        motion_dir = project_path(meta["motion_dir"])
    material = material_from_config(config)
    body_model = None
    if local_body_feature_mode != "none":
        body_model = build_ncs_model(config, topology.edge_count)

    out_dir.mkdir(parents=True, exist_ok=True)
    if alpha_dir:
        alpha_dir.mkdir(parents=True, exist_ok=True)

    items = list(garment_data.items())
    if args.sequence_name:
        selected = {Path(name).stem for name in args.sequence_name}
        items = [(name, value) for name, value in items if Path(name).stem in selected]
    if args.max_sequences is not None:
        items = items[: args.max_sequences]
    if not items:
        raise FileNotFoundError("No matching .npy sequence names in garment-dir")
    for name, positions in items:
        features = load_or_prepare_features_np(
            name,
            positions,
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
        input_rest = input_rest_data.get(name)
        threshold, alpha = predict_threshold(
            model,
            features,
            absolute=args.absolute,
            alpha_cutoff=args.alpha_cutoff,
            node_position_mode=node_position_mode,
            output_mode=output_mode,
            motion_features=motion_features,
            local_body_features=local_body_features,
            stage_feature=stage_feature,
            input_rest=input_rest,
            drive_threshold=drive_threshold,
            drive_window=args.drive_window,
            drive_min_frames=args.drive_min_frames,
        )
        if args.rest_scale != 1.0:
            threshold = (threshold * float(args.rest_scale)).astype(np.float32)
        np.save(out_dir / f"{name}.npy", threshold)
        if alpha_dir:
            np.save(alpha_dir / f"{name}.npy", alpha)
        print("Saved P-Net threshold:", out_dir / f"{name}.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict EPNet plasticity files with TensorFlow P-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=r"results\checkpoints\pnet\pnet.weights.h5")
    parser.add_argument("--garment-dir", default=r"results\elasticity\iteration_0\garment")
    parser.add_argument("--input-rest-dir", default=None)
    parser.add_argument("--motion-dir", default=None)
    parser.add_argument("--motion-feature-mode", choices=("none", "summary"), default=None)
    parser.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default=None)
    parser.add_argument("--stage-feature-mode", choices=("none", "scalar"), default=None)
    parser.add_argument("--stage-value", type=float, default=None)
    parser.add_argument("--stage-normalizer", type=float, default=None)
    parser.add_argument("--out-dir", default=r"results\rest")
    parser.add_argument("--alpha-dir", default=r"results\alpha")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--absolute", action="store_true", help="Save abs(theta_rest). Default preserves sign.")
    parser.add_argument("--alpha-cutoff", type=float, default=0.05)
    parser.add_argument("--drive-threshold", type=float, default=None)
    parser.add_argument("--drive-window", type=int, default=1)
    parser.add_argument("--drive-min-frames", type=int, default=1)
    parser.add_argument("--rest-scale", type=float, default=1.0)
    parser.add_argument(
        "--node-position-mode",
        choices=("absolute", "displacement", "none"),
        default=None,
    )
    parser.add_argument("--output-mode", choices=("alpha", "direct", "mask_direct", "delta_direct"), default=None)
    parser.add_argument("--topology-cache", default=None)
    parser.add_argument("--rebuild-topology", action="store_true")
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
