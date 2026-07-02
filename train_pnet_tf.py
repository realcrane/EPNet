from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
from tqdm import tqdm

from global_vars import ROOT_DIR
from model.build import build_ncs_model
from epnet.tf_features import load_or_prepare_features_np, load_sorted_npy, wrap_to_pi_np
from epnet.tf_model import make_pnet_model
from epnet.topology import get_topology
from utils.config import MainConfig
from utils.rotation import axis_angle_to_quat


PROJECT_ROOT = Path(ROOT_DIR)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def node_feature_count(position_mode: str) -> int:
    if position_mode == "absolute":
        return 10
    if position_mode == "displacement":
        return 7
    if position_mode == "none":
        return 4
    raise ValueError(f"Unsupported node position mode: {position_mode}")


def motion_feature_count(motion_mode: str, input_joints: list[int]) -> int:
    if motion_mode == "none":
        return 0
    if motion_mode == "summary":
        pose_dim = len(input_joints) * 3
        return pose_dim * 4 + 10
    raise ValueError(f"Unsupported motion feature mode: {motion_mode}")


def local_body_feature_count(local_body_mode: str) -> int:
    if local_body_mode == "none":
        return 0
    if local_body_mode == "nearest":
        return 16
    raise ValueError(f"Unsupported local body feature mode: {local_body_mode}")


def stage_feature_count(stage_mode: str) -> int:
    if stage_mode == "none":
        return 0
    if stage_mode == "scalar":
        return 3
    raise ValueError(f"Unsupported stage feature mode: {stage_mode}")


def finite_features(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(array, -10.0, 10.0).astype(np.float32)


def finite_tensor(tensor: tf.Tensor, limit: float = 10.0) -> tf.Tensor:
    tensor = tf.where(tf.math.is_finite(tensor), tensor, tf.zeros_like(tensor))
    return tf.clip_by_value(tensor, -limit, limit)


def make_stage_feature(stage_value: float, stage_normalizer: float, stage_mode: str) -> np.ndarray | None:
    if stage_mode == "none":
        return None
    if stage_mode != "scalar":
        raise ValueError(f"Unsupported stage feature mode: {stage_mode}")
    normalized = np.float32(stage_value / max(stage_normalizer, 1e-6))
    return np.array(
        [normalized, np.sin(np.pi * normalized), np.cos(np.pi * normalized)],
        dtype=np.float32,
    )


def align_time_count(array: np.ndarray, t_count: int) -> np.ndarray:
    if array.shape[0] >= t_count:
        return array[:t_count]
    pad = np.broadcast_to(array[-1:], (t_count - array.shape[0], *array.shape[1:]))
    return np.concatenate([array, pad], axis=0)


def future_cumulative_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    future_max = np.maximum.accumulate(values[::-1], axis=0)[::-1]
    future_sum = np.cumsum(values[::-1], axis=0)[::-1]
    counts = np.arange(values.shape[0], 0, -1, dtype=np.float32)[:, None]
    return future_sum / counts, future_max


def load_motion_features_np(
    motion_dir: Path | None,
    sequence_name: str,
    t_count: int,
    input_joints: list[int],
    motion_mode: str,
) -> np.ndarray | None:
    if motion_mode == "none":
        return None
    if motion_dir is None:
        raise ValueError("--motion-dir is required when --motion-feature-mode is not 'none'")

    path = Path(motion_dir) / f"{Path(sequence_name).stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {path}")
    data = np.load(path)
    poses = data["poses"].astype(np.float32).reshape(-1, 24, 3)[:, input_joints].reshape(-1, len(input_joints) * 3)
    trans = data["trans"].astype(np.float32)
    poses = align_time_count(poses, t_count)
    trans = align_time_count(trans, t_count)

    pose_vel = np.zeros_like(poses)
    trans_vel = np.zeros_like(trans)
    pose_vel[1:] = poses[1:] - poses[:-1]
    trans_vel[1:] = trans[1:] - trans[:-1]
    future_pose_vel_mean, future_pose_vel_max = future_cumulative_stats(np.abs(pose_vel))
    future_speed = np.linalg.norm(trans_vel, axis=-1, keepdims=True)
    future_speed_mean, future_speed_max = future_cumulative_stats(future_speed)
    time_coord = np.linspace(0.0, 1.0, t_count, dtype=np.float32)[:, None]
    motion = np.concatenate(
        [
            poses,
            pose_vel,
            future_pose_vel_mean,
            future_pose_vel_max,
            trans,
            trans_vel,
            future_speed_mean,
            future_speed_max,
            time_coord,
            1.0 - time_coord,
        ],
        axis=-1,
    )
    return finite_features(motion)


def load_body_positions_np(body_model, motion_dir: Path, sequence_name: str, t_count: int) -> np.ndarray:
    path = Path(motion_dir) / f"{Path(sequence_name).stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {path}")
    data = np.load(path)
    poses = data["poses"].astype(np.float32).reshape(-1, 24, 3)
    poses = axis_angle_to_quat(poses).astype(np.float32)
    poses[:, 22:24, :] = 0.0
    trans = data["trans"].astype(np.float32)
    poses = align_time_count(poses, t_count)
    trans = align_time_count(trans, t_count)
    _, matrices = body_model.call_inputs(tf.convert_to_tensor(poses[None]), tf.convert_to_tensor(trans[None]))
    body = body_model.lbs_body(body_model.body.vertices, matrices)
    body_np = np.asarray(body[0], dtype=np.float32)
    if body_np.shape[0] < t_count:
        pad_count = t_count - body_np.shape[0]
        body_np = np.concatenate([np.broadcast_to(body_np[:1], (pad_count, *body_np.shape[1:])), body_np], axis=0)
    return finite_features(body_np[:t_count])


def nearest_body_features_np(mid_pos: np.ndarray, body_pos: np.ndarray) -> np.ndarray:
    t_count, hinge_count, _ = mid_pos.shape
    local = np.empty((t_count, hinge_count, 16), dtype=np.float32)
    body_vel = np.zeros_like(body_pos)
    body_vel[1:] = body_pos[1:] - body_pos[:-1]
    hinge_vel = np.zeros_like(mid_pos)
    hinge_vel[1:] = mid_pos[1:] - mid_pos[:-1]

    for t in range(t_count):
        body_t = body_pos[t]
        dist, nearest = cKDTree(body_t).query(mid_pos[t], k=1)
        nearest_body = body_t[nearest]
        rel = mid_pos[t] - nearest_body
        bv = body_vel[t, nearest]
        hv = hinge_vel[t]
        local[t] = np.concatenate(
            [
                rel,
                np.abs(rel),
                np.maximum(dist.astype(np.float32), 1e-6)[:, None],
                bv,
                hv,
                hv - bv,
            ],
            axis=-1,
        )
    return finite_features(local)


def local_body_cache_path(cache_dir: Path, sequence_name: str) -> Path:
    return Path(cache_dir) / f"{sequence_name}.local_body.npz"


def load_local_body_features_np(
    body_model,
    motion_dir: Path | None,
    sequence_name: str,
    features: dict[str, np.ndarray],
    t_count: int,
    local_body_mode: str,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> np.ndarray | None:
    if local_body_mode == "none":
        return None
    if local_body_mode != "nearest":
        raise ValueError(f"Unsupported local body feature mode: {local_body_mode}")
    if motion_dir is None:
        raise ValueError("--motion-dir is required when --local-body-feature-mode is not 'none'")
    if body_model is None:
        raise ValueError("A body model is required when --local-body-feature-mode is not 'none'")

    if cache_dir is not None:
        path = local_body_cache_path(cache_dir, sequence_name)
        if path.is_file() and not rebuild:
            data = np.load(path)
            local = finite_features(data["local_body_features"])
            return local[:t_count]

    mid_pos = features["mid_pos_cache"][:t_count].astype(np.float32)
    body_pos = load_body_positions_np(body_model, motion_dir, sequence_name, t_count)
    local = nearest_body_features_np(mid_pos, body_pos)
    if cache_dir is not None:
        path = local_body_cache_path(cache_dir, sequence_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, local_body_features=local)
    return finite_features(local)


def make_node_features(
    theta_seq: np.ndarray,
    rest_prev: np.ndarray,
    rest_curr: np.ndarray,
    features: dict[str, np.ndarray],
    t: int,
    position_mode: str,
    motion_features: np.ndarray | None = None,
    local_body_features: np.ndarray | None = None,
    stage_feature: np.ndarray | None = None,
) -> np.ndarray:
    parts = [theta_seq[t - 1], theta_seq[t], rest_prev, rest_curr]
    if position_mode == "absolute":
        parts.extend([features["mid_pos_cache"][t], features["mid_rest_static"]])
    elif position_mode == "displacement":
        parts.append(features["mid_pos_cache"][t] - features["mid_rest_static"])
    elif position_mode != "none":
        raise ValueError(f"Unsupported node position mode: {position_mode}")
    if motion_features is not None:
        hinge_count = theta_seq.shape[1]
        parts.append(np.broadcast_to(motion_features[t][None], (hinge_count, motion_features.shape[-1])))
    if local_body_features is not None:
        parts.append(local_body_features[t])
    if stage_feature is not None:
        hinge_count = theta_seq.shape[1]
        parts.append(np.broadcast_to(stage_feature[None], (hinge_count, stage_feature.shape[-1])))
    return finite_features(np.concatenate(parts, axis=-1))


def make_time_sample(
    theta_seq: np.ndarray,
    target: np.ndarray,
    input_rest: np.ndarray | None,
    features: dict[str, np.ndarray],
    t: int,
    noise_std: float,
    rest_input_mode: str,
    node_position_mode: str,
    motion_features: np.ndarray | None,
    local_body_features: np.ndarray | None,
    stage_feature: np.ndarray | None,
):
    if rest_input_mode == "zero":
        rest_prev = np.zeros_like(target[t - 1])
        rest_curr = np.zeros_like(target[t])
    elif rest_input_mode == "target":
        source = target if input_rest is None else input_rest
        rest_prev = source[t - 1]
        rest_curr = source[t]
    elif rest_input_mode == "previous":
        if input_rest is None:
            rest_prev = np.zeros_like(target[t - 1])
            rest_curr = np.zeros_like(target[t])
        else:
            rest_prev = input_rest[t - 1]
            rest_curr = input_rest[t]
    else:
        raise ValueError(f"Unsupported rest input mode: {rest_input_mode}")

    rest_prev = rest_prev + noise_std * np.random.randn(*rest_prev.shape).astype(np.float32)
    rest_curr = rest_curr + noise_std * np.random.randn(*rest_curr.shape).astype(np.float32)
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

    target_delta = (target[t + 1] - target[t]).astype(np.float32)
    drive = wrap_to_pi_np(theta_seq[t] - target[t]).astype(np.float32)
    alpha_gt = np.clip(
        target_delta * drive / np.maximum(drive**2, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    return node_features, features["edge_feat_cache"][t], alpha_gt, target_delta, drive, target[t]


def make_graph_batch(samples, base_edge_index: np.ndarray):
    node_parts, edge_parts, alpha_parts, delta_parts, drive_parts, target_parts = zip(*samples)
    node_count = node_parts[0].shape[0]
    edge_indices = []
    for batch_idx in range(len(samples)):
        edge_indices.append(base_edge_index + batch_idx * node_count)
    return (
        tf.convert_to_tensor(np.concatenate(node_parts, axis=0), dtype=tf.float32),
        tf.convert_to_tensor(np.concatenate(edge_indices, axis=1), dtype=tf.int32),
        tf.convert_to_tensor(np.concatenate(edge_parts, axis=0), dtype=tf.float32),
        tf.convert_to_tensor(np.concatenate(alpha_parts, axis=0), dtype=tf.float32),
        tf.convert_to_tensor(np.concatenate(delta_parts, axis=0), dtype=tf.float32),
        tf.convert_to_tensor(np.concatenate(drive_parts, axis=0), dtype=tf.float32),
        tf.convert_to_tensor(np.concatenate(target_parts, axis=0), dtype=tf.float32),
    )


def pnet_loss(
    model,
    node_features,
    edge_index,
    edge_features,
    alpha_gt,
    target_delta,
    drive,
    active_weight,
    active_threshold,
    drive_threshold,
    inactive_weight,
    direct_target,
    output_mode,
    training,
):
    raw_pred = finite_tensor(tf.cast(model((node_features, edge_index, edge_features), training=training), tf.float32), 10.0)
    alpha_gt = finite_tensor(tf.cast(alpha_gt, tf.float32), 1.0)
    target_delta = finite_tensor(tf.cast(target_delta, tf.float32), 1.0)
    drive = finite_tensor(tf.cast(drive, tf.float32), 1.0)
    direct_target = finite_tensor(tf.cast(direct_target, tf.float32), 1.0)
    if output_mode == "mask_direct":
        mask_logit = raw_pred[:, :1]
        rest_pred = raw_pred[:, 1:2]
        active = tf.cast(tf.abs(direct_target) > active_threshold, tf.float32)
        active_den = tf.maximum(tf.reduce_sum(active), 1.0)
        inactive = 1.0 - active
        inactive_den = tf.maximum(tf.reduce_sum(inactive), 1.0)
        mask_loss_active = tf.reduce_sum(
            active * tf.nn.sigmoid_cross_entropy_with_logits(labels=active, logits=mask_logit)
        ) / active_den
        mask_loss_inactive = tf.reduce_sum(
            inactive * tf.nn.sigmoid_cross_entropy_with_logits(labels=active, logits=mask_logit)
        ) / inactive_den
        rest_loss_active = tf.reduce_sum(active * tf.abs(rest_pred - direct_target)) / active_den
        rest_loss_inactive = tf.reduce_mean(inactive * tf.abs(rest_pred))
        return (
            active_weight * (mask_loss_active + rest_loss_active)
            + inactive_weight * (mask_loss_inactive + rest_loss_inactive)
        )
    if output_mode == "direct":
        active = tf.cast(tf.abs(direct_target) > active_threshold, tf.float32)
        active_den = tf.maximum(tf.reduce_sum(active), 1.0)
        loss_active = tf.reduce_sum(active * tf.abs(raw_pred - direct_target)) / active_den
        inactive = 1.0 - active
        loss_inactive = tf.reduce_mean(inactive * tf.abs(raw_pred))
        return active_weight * loss_active + inactive_weight * loss_inactive
    if output_mode == "delta_direct":
        drive_active = tf.cast(tf.abs(drive) > drive_threshold, tf.float32)
        target_delta = target_delta * drive_active
        active = tf.cast(tf.abs(target_delta) > active_threshold, tf.float32)
        active_den = tf.maximum(tf.reduce_sum(active), 1.0)
        loss_active = tf.reduce_sum(active * tf.abs(raw_pred - target_delta)) / active_den
        inactive = 1.0 - active
        loss_inactive = tf.reduce_mean(inactive * tf.abs(raw_pred))
        return active_weight * loss_active + inactive_weight * loss_inactive

    alpha_hat = tf.sigmoid(raw_pred)
    active = tf.cast(tf.abs(target_delta) > active_threshold, tf.float32)
    if active_weight > 0.0:
        active_den = tf.maximum(tf.reduce_sum(active), 1.0)
        loss_alpha = tf.reduce_sum(active * tf.abs(alpha_hat - alpha_gt)) / active_den
        loss_delta = tf.reduce_sum(active * tf.abs(alpha_hat * drive - target_delta)) / active_den
        inactive = 1.0 - active
        loss_inactive = tf.reduce_mean(inactive * tf.abs(alpha_hat))
        return active_weight * (loss_alpha + 0.05 * loss_delta) + inactive_weight * loss_inactive

    loss_alpha = tf.reduce_mean(tf.abs(alpha_hat - alpha_gt))
    loss_delta = tf.reduce_mean(tf.abs(alpha_hat * drive - target_delta))
    return loss_alpha + 0.05 * loss_delta


@tf.function
def evaluate_graph_batch(
    model,
    node_features,
    edge_index,
    edge_features,
    alpha_gt,
    target_delta,
    drive,
    active_weight,
    active_threshold,
    drive_threshold,
    inactive_weight,
    direct_target,
    output_mode,
):
    return pnet_loss(
        model,
        node_features,
        edge_index,
        edge_features,
        alpha_gt,
        target_delta,
        drive,
        active_weight,
        active_threshold,
        drive_threshold,
        inactive_weight,
        direct_target,
        output_mode,
        training=False,
    )


@tf.function
def train_graph_batch(
    model,
    optimizer,
    node_features,
    edge_index,
    edge_features,
    alpha_gt,
    target_delta,
    drive,
    active_weight,
    active_threshold,
    drive_threshold,
    inactive_weight,
    direct_target,
    output_mode,
):
    with tf.GradientTape() as tape:
        loss = pnet_loss(
            model,
            node_features,
            edge_index,
            edge_features,
            alpha_gt,
            target_delta,
            drive,
            active_weight,
            active_threshold,
            drive_threshold,
            inactive_weight,
            direct_target,
            output_mode,
            training=True,
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [
        None if grad is None else tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        for grad in gradients
    ]
    gradients, _ = tf.clip_by_global_norm(gradients, 0.1)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_one_sequence(
    model,
    optimizer,
    features: dict[str, np.ndarray],
    target_np: np.ndarray,
    input_rest_np: np.ndarray | None,
    noise_std: float,
    time_batch_size: int,
    active_weight: float,
    active_threshold: float,
    drive_threshold: float,
    inactive_weight: float,
    rest_input_mode: str,
    node_position_mode: str,
    motion_features: np.ndarray | None,
    local_body_features: np.ndarray | None,
    stage_feature: np.ndarray | None,
    output_mode: str,
    progress_desc: str | None = None,
) -> list[float]:
    theta_seq = features["theta_seq"].astype(np.float32)
    target = target_np.astype(np.float32)
    if target.ndim == 2:
        target = target[..., None]
    input_rest = None
    if input_rest_np is not None:
        input_rest = input_rest_np.astype(np.float32)
        if input_rest.ndim == 2:
            input_rest = input_rest[..., None]

    t_count = min(theta_seq.shape[0], target.shape[0])
    if input_rest is not None:
        t_count = min(t_count, input_rest.shape[0])
    if t_count < 3:
        return []
    theta_seq = theta_seq[:t_count]
    target = target[:t_count]
    if input_rest is not None:
        input_rest = input_rest[:t_count]

    losses = []
    order = np.arange(1, t_count - 1)
    np.random.shuffle(order)
    time_batch_size = max(1, int(time_batch_size))
    ranges = range(0, len(order), time_batch_size)
    if progress_desc:
        ranges = tqdm(ranges, desc=progress_desc, leave=False)
    for start in ranges:
        batch_ts = order[start : start + time_batch_size]
        samples = [
            make_time_sample(
                theta_seq,
                target,
                input_rest,
                features,
                int(t),
                noise_std,
                rest_input_mode,
                node_position_mode,
                motion_features,
                local_body_features,
                stage_feature,
            )
            for t in batch_ts
        ]
        batch = make_graph_batch(samples, features["edge_index"])
        loss = train_graph_batch(
            model,
            optimizer,
            *batch[:-1],
            active_weight,
            tf.constant(active_threshold, dtype=tf.float32),
            tf.constant(drive_threshold, dtype=tf.float32),
            tf.constant(inactive_weight, dtype=tf.float32),
            batch[-1],
            output_mode,
        )
        losses.append(float(loss.numpy()))
    return losses


def evaluate_one_sequence(
    model,
    features: dict[str, np.ndarray],
    target_np: np.ndarray,
    input_rest_np: np.ndarray | None,
    time_batch_size: int,
    active_weight: float,
    active_threshold: float,
    drive_threshold: float,
    inactive_weight: float,
    rest_input_mode: str,
    node_position_mode: str,
    motion_features: np.ndarray | None,
    local_body_features: np.ndarray | None,
    stage_feature: np.ndarray | None,
    output_mode: str,
) -> list[float]:
    theta_seq = features["theta_seq"].astype(np.float32)
    target = target_np.astype(np.float32)
    if target.ndim == 2:
        target = target[..., None]
    input_rest = None
    if input_rest_np is not None:
        input_rest = input_rest_np.astype(np.float32)
        if input_rest.ndim == 2:
            input_rest = input_rest[..., None]

    t_count = min(theta_seq.shape[0], target.shape[0])
    if input_rest is not None:
        t_count = min(t_count, input_rest.shape[0])
    if t_count < 3:
        return []
    theta_seq = theta_seq[:t_count]
    target = target[:t_count]
    if input_rest is not None:
        input_rest = input_rest[:t_count]

    losses = []
    order = np.arange(1, t_count - 1)
    time_batch_size = max(1, int(time_batch_size))
    for start in range(0, len(order), time_batch_size):
        batch_ts = order[start : start + time_batch_size]
        samples = [
            make_time_sample(
                theta_seq,
                target,
                input_rest,
                features,
                int(t),
                noise_std=0.0,
                rest_input_mode=rest_input_mode,
                node_position_mode=node_position_mode,
                motion_features=motion_features,
                local_body_features=local_body_features,
                stage_feature=stage_feature,
            )
            for t in batch_ts
        ]
        batch = make_graph_batch(samples, features["edge_index"])
        loss = evaluate_graph_batch(
            model,
            *batch[:-1],
            active_weight,
            tf.constant(active_threshold, dtype=tf.float32),
            tf.constant(drive_threshold, dtype=tf.float32),
            tf.constant(inactive_weight, dtype=tf.float32),
            batch[-1],
            output_mode,
        )
        losses.append(float(loss.numpy()))
    return losses


def material_from_config(config: MainConfig) -> dict:
    return {
        "timestep": config.time_step,
        "bending_coeff": config.loss.bending,
        "lame_mu": getattr(config.loss.cloth, "stretch", 10.0),
        "lame_lambda": getattr(config.loss.cloth, "shear", 1.0),
    }


def parse_stage_path_args(items: list[str] | None, default_stage: float, default_path: str | None) -> list[tuple[float, Path, str]]:
    if not items:
        if default_path is None:
            return []
        return [(default_stage, project_path(default_path), str(Path(default_path)))]

    parsed = []
    for item in items:
        if ":" not in item:
            raise ValueError("Stage paths must use the form STAGE:PATH, for example 3:results\\...\\threshold")
        stage_text, path_text = item.split(":", 1)
        parsed.append((float(stage_text), project_path(path_text), path_text))
    return parsed


def stage_key(stage: float) -> str:
    return f"{stage:g}"


def stage_cache_name(sequence_name: str, stage: float, use_stage_garment: bool) -> str:
    if not use_stage_garment:
        return sequence_name
    return f"{sequence_name}_stage{stage_key(stage)}"


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    garment_dir = project_path(args.garment_dir)
    motion_dir = project_path(args.motion_dir) if args.motion_dir else None
    feature_cache_dir = project_path(args.feature_cache_dir) if args.feature_cache_dir else None
    topology_cache = project_path(args.topology_cache) if args.topology_cache else None

    config = MainConfig(str(config_path))
    topology = get_topology(config, cache_path=topology_cache, rebuild=args.rebuild_topology)
    default_garment_data = load_sorted_npy(garment_dir)
    target_specs = parse_stage_path_args(args.stage_target_dir, args.stage_value, args.target_dir)
    garment_specs = parse_stage_path_args(args.stage_garment_dir, args.stage_value, None)
    input_rest_specs = parse_stage_path_args(args.stage_input_rest_dir, args.stage_value, None)
    garment_by_stage = {stage_key(stage): (load_sorted_npy(path), original) for stage, path, original in garment_specs}
    input_rest_by_stage = {stage_key(stage): (load_sorted_npy(path), original) for stage, path, original in input_rest_specs}
    target_data_by_stage = []
    names = set(default_garment_data)
    for stage, path, original in target_specs:
        key = stage_key(stage)
        target_data = load_sorted_npy(path)
        garment_data, garment_original = garment_by_stage.get(key, (default_garment_data, str(Path(args.garment_dir))))
        input_rest_data, input_rest_original = input_rest_by_stage.get(key, (None, None))
        names &= set(garment_data) & set(target_data)
        if input_rest_data is not None:
            names &= set(input_rest_data)
        target_data_by_stage.append((stage, garment_data, target_data, input_rest_data, original, garment_original, input_rest_original))
    names = sorted(names)
    if args.sequence_name:
        selected = {Path(name).stem for name in args.sequence_name}
        names = [name for name in names if Path(name).stem in selected]
    if args.max_sequences is not None:
        names = names[: args.max_sequences]
    if not names:
        raise FileNotFoundError("No matching .npy sequence names between garment-dir and target-dir")
    validation_names = []
    if args.validation_sequence_name:
        selected = {Path(name).stem for name in args.validation_sequence_name}
        validation_names = [name for name in names if Path(name).stem in selected]
        names = [name for name in names if Path(name).stem not in selected]
        if not validation_names:
            raise FileNotFoundError("No validation sequence names matched the P-Net data")
        if not names:
            raise ValueError("Validation split consumed all training sequences")

    model, model_config = make_pnet_model(
        {
            "n_nodefeatures": node_feature_count(args.node_position_mode)
            + motion_feature_count(args.motion_feature_mode, config.body.input_joints)
            + local_body_feature_count(args.local_body_feature_mode)
            + stage_feature_count(args.stage_feature_mode),
            "output_size": 2 if args.output_mode == "mask_direct" else 1,
        }
    )
    if args.load:
        load_path = project_path(args.load)
        model.load_weights(str(load_path))
        print("Loaded P-Net checkpoint:", load_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    material = material_from_config(config)
    body_model = None
    if args.local_body_feature_mode != "none":
        body_model = build_ncs_model(config, topology.edge_count)

    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_out = project_path(args.best_out) if args.best_out else out.with_name(f"best_{out.name}")
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale_epochs = 0
    all_losses = []
    for epoch in range(args.epochs):
        epoch_losses = []
        sequence_bar = tqdm(names, desc=f"[train-pnet-tf] epoch {epoch + 1}/{args.epochs}", leave=True)
        for name in sequence_bar:
            sequence_bar.set_postfix_str(name[:40])
            stage_order = list(target_data_by_stage)
            np.random.shuffle(stage_order)
            for stage_value, garment_data, target_data, input_rest_data, *_ in stage_order:
                cache_name = stage_cache_name(name, stage_value, bool(args.stage_garment_dir))
                features = load_or_prepare_features_np(
                    cache_name,
                    garment_data[name],
                    topology,
                    material=material,
                    cache_dir=feature_cache_dir,
                    rebuild=args.rebuild_feature_cache,
                )
                input_rest_np = None if input_rest_data is None else input_rest_data[name]
                t_count = min(features["theta_seq"].shape[0], target_data[name].shape[0])
                if input_rest_np is not None:
                    t_count = min(t_count, input_rest_np.shape[0])
                motion_features = load_motion_features_np(
                    motion_dir,
                    name,
                    t_count,
                    config.body.input_joints,
                    args.motion_feature_mode,
                )
                local_body_features = load_local_body_features_np(
                    body_model,
                    motion_dir,
                    name,
                    features,
                    t_count,
                    args.local_body_feature_mode,
                    cache_dir=feature_cache_dir,
                    rebuild=args.rebuild_feature_cache,
                )
                stage_feature = make_stage_feature(stage_value, args.stage_normalizer, args.stage_feature_mode)
                epoch_losses.extend(
                    train_one_sequence(
                        model,
                        optimizer,
                        features,
                        target_data[name],
                        input_rest_np,
                        args.noise_std,
                        args.time_batch_size,
                        args.active_weight,
                        args.active_threshold,
                        args.drive_threshold,
                        args.inactive_weight,
                        args.rest_input_mode,
                        args.node_position_mode,
                        motion_features,
                        local_body_features,
                        stage_feature,
                        args.output_mode,
                        progress_desc=f"{name[:18]} s{stage_value:g}",
                    )
                )
        all_losses.extend(epoch_losses)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_losses = []
        for name in validation_names:
            for stage_value, garment_data, target_data, input_rest_data, *_ in target_data_by_stage:
                cache_name = stage_cache_name(name, stage_value, bool(args.stage_garment_dir))
                features = load_or_prepare_features_np(
                    cache_name,
                    garment_data[name],
                    topology,
                    material=material,
                    cache_dir=feature_cache_dir,
                    rebuild=args.rebuild_feature_cache,
                )
                input_rest_np = None if input_rest_data is None else input_rest_data[name]
                t_count = min(features["theta_seq"].shape[0], target_data[name].shape[0])
                if input_rest_np is not None:
                    t_count = min(t_count, input_rest_np.shape[0])
                motion_features = load_motion_features_np(
                    motion_dir,
                    name,
                    t_count,
                    config.body.input_joints,
                    args.motion_feature_mode,
                )
                local_body_features = load_local_body_features_np(
                    body_model,
                    motion_dir,
                    name,
                    features,
                    t_count,
                    args.local_body_feature_mode,
                    cache_dir=feature_cache_dir,
                    rebuild=args.rebuild_feature_cache,
                )
                stage_feature = make_stage_feature(stage_value, args.stage_normalizer, args.stage_feature_mode)
                val_losses.extend(
                    evaluate_one_sequence(
                        model,
                        features,
                        target_data[name],
                        input_rest_np,
                        args.time_batch_size,
                        args.active_weight,
                        args.active_threshold,
                        args.drive_threshold,
                        args.inactive_weight,
                        args.rest_input_mode,
                        args.node_position_mode,
                        motion_features,
                        local_body_features,
                        stage_feature,
                        args.output_mode,
                    )
                )
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(
            f"[train-pnet-tf] epoch={epoch} mean_loss={mean_loss:.6f} "
            f"val_loss={val_loss:.6f} steps={len(epoch_losses)} val_steps={len(val_losses)}"
        )
        if val_losses and val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            stale_epochs = 0
            model.save_weights(str(best_out))
            print(f"[train-pnet-tf] saved best checkpoint: {best_out} val_loss={best_val:.6f}")
        elif val_losses:
            stale_epochs += 1
            if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
                print(
                    f"[train-pnet-tf] early stopping at epoch={epoch}; "
                    f"best_val_loss={best_val:.6f}"
                )
                break

    model.save_weights(str(out))
    meta = {
        "model_config": model_config,
        "config": str(Path(args.config)),
        "garment_dir": str(Path(args.garment_dir)),
        "target_dir": str(Path(args.target_dir)),
        "stage_targets": [
            {
                "stage": stage,
                "target_dir": target_original,
                "garment_dir": garment_original,
                "input_rest_dir": input_rest_original,
            }
            for stage, _, _, _, target_original, garment_original, input_rest_original in target_data_by_stage
        ],
        "stage_feature_mode": args.stage_feature_mode,
        "stage_normalizer": args.stage_normalizer,
        "stage_value": args.stage_value,
        "validation_sequences": args.validation_sequence_name or [],
        "rest_input_mode": args.rest_input_mode,
        "node_position_mode": args.node_position_mode,
        "motion_feature_mode": args.motion_feature_mode,
        "local_body_feature_mode": args.local_body_feature_mode,
        "motion_dir": str(Path(args.motion_dir)) if args.motion_dir else None,
        "output_mode": args.output_mode,
        "drive_threshold": args.drive_threshold,
        "load": str(Path(args.load)) if args.load else None,
        "best_out": str(best_out),
        "best_val_loss": best_val if np.isfinite(best_val) else None,
        "early_stopping_patience": args.early_stopping_patience,
        "source": "epnet_threshold_tf",
        "loss": all_losses,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    best_out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved P-Net checkpoint:", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow P-Net from EPNet garment and threshold .npy files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--garment-dir", default=r"results\elasticity\iteration_0\garment")
    parser.add_argument("--target-dir", default=r"results\elasticity\iteration_0\threshold")
    parser.add_argument(
        "--stage-garment-dir",
        action="append",
        default=None,
        help="Optional multi-stage garment input in STAGE:PATH form.",
    )
    parser.add_argument(
        "--stage-target-dir",
        action="append",
        default=None,
        help="Optional multi-stage target in STAGE:PATH form. Repeat for rest1/rest2/rest3 curriculum supervision.",
    )
    parser.add_argument(
        "--stage-input-rest-dir",
        action="append",
        default=None,
        help="Optional previous-rest input in STAGE:PATH form. Used when --rest-input-mode target.",
    )
    parser.add_argument("--stage-feature-mode", choices=("none", "scalar"), default="none")
    parser.add_argument("--stage-value", type=float, default=1.0)
    parser.add_argument("--stage-normalizer", type=float, default=3.0)
    parser.add_argument("--motion-dir", default=None)
    parser.add_argument("--motion-feature-mode", choices=("none", "summary"), default="none")
    parser.add_argument("--local-body-feature-mode", choices=("none", "nearest"), default="none")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--drive-threshold", type=float, default=0.0)
    parser.add_argument("--inactive-weight", type=float, default=1e-3)
    parser.add_argument("--rest-input-mode", choices=("target", "zero", "previous"), default="target")
    parser.add_argument(
        "--node-position-mode",
        choices=("absolute", "displacement", "none"),
        default="absolute",
    )
    parser.add_argument("--output-mode", choices=("alpha", "direct", "mask_direct", "delta_direct"), default="alpha")
    parser.add_argument("--load", default=None)
    parser.add_argument("--out", default=r"results\checkpoints\pnet\pnet.weights.h5")
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--validation-sequence-name", action="append", default=None)
    parser.add_argument("--topology-cache", default=None)
    parser.add_argument("--rebuild-topology", action="store_true")
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--time-batch-size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
