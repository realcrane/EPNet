from __future__ import annotations

import json
import os
from math import ceil
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence

from epnet.global_vars import BODY_DIR, DATA_DIR, TXT_DIR
from model.sequence import PoseSequence


class ElasticityDataset(Sequence):
    """Dataset for elastic NCS training and prediction."""

    def __init__(
        self,
        config,
        edge_count: int,
        mode: str = "test",
        max_sequences: int | None = None,
        sequence_names: list[str] | None = None,
        target_input_noise_std: float = 0.0,
        target_noise_as_target: bool = False,
        target_noise_smoothing_steps: int = 0,
        warmup_frames: int = 0,
        edge_neighbors: list[np.ndarray] | None = None,
        target_noise_edge_mask: np.ndarray | None = None,
        rest_zero_rehearsal_probability: float = 0.0,
    ) -> None:
        if mode not in {"train", "validation", "test"}:
            raise ValueError(
                "mode must be one of {'train', 'validation', 'test'}, got "
                f"{mode!r}"
            )

        self.config = config
        self.edge_count = int(edge_count)
        self.mode = mode
        self.batch_size = int(config.experiment.elasticity_batch_size)
        self.reflect_probability = float(config.experiment.reflect_probability)
        self.txt_path = Path(TXT_DIR) / config.data.dataset / getattr(config.data, mode)
        self.files_name = self._read_sequence_names(max_sequences, sequence_names)
        self.sequence_paths = [
            Path(DATA_DIR) / config.data.dataset / name for name in self.files_name
        ]
        self.sequences = [PoseSequence(str(path)) for path in self.sequence_paths]
        self.seq_idx = np.arange(len(self.sequences))
        self.seq_duration = np.array([seq.duration for seq in self.sequences])
        self.seq_frames = np.array([seq.num_frames for seq in self.sequences])
        self.targets: list[np.ndarray] | None = None
        self.is_zero_target = True
        self.target_input_noise_std = float(target_input_noise_std)
        self.target_noise_as_target = bool(target_noise_as_target)
        self.target_noise_smoothing_steps = int(target_noise_smoothing_steps)
        self.warmup_frames = max(0, int(warmup_frames))
        self.rest_zero_rehearsal_probability = float(rest_zero_rehearsal_probability)
        self.edge_neighbor_indices, self.edge_neighbor_mask = self._make_edge_neighbors(
            edge_neighbors
        )
        self.target_noise_edge_mask = self._make_target_noise_edge_mask(
            target_noise_edge_mask
        )
        self._initial_target_noise_cache: dict[tuple[int, int], np.ndarray] = {}

        self._read_skeleton()
        self._make_reflection_map()
        self._make_sample_list()

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_time_steps(self) -> int:
        return int(self.config.num_time_steps)

    @property
    def skeleton_shape(self) -> tuple[int, int]:
        return (self.num_joints, 4)

    @property
    def sample_poses_shape(self) -> list[int]:
        return [self.num_time_steps, *self.skeleton_shape]

    @property
    def sample_trans_shape(self) -> list[int]:
        return [self.num_time_steps, 3]

    @property
    def sample_plasticity_shape(self) -> list[int]:
        return [self.num_time_steps - 2, self.edge_count, 1]

    @property
    def sample_phase_shape(self) -> list[int]:
        return [self.num_time_steps - 2, 2]

    def expected_sequence_names(self) -> list[str]:
        return [Path(name).stem for name in self.files_name]

    def time_steps(self, time: float) -> np.ndarray:
        return time + np.arange(1 - self.num_time_steps, 1) * self.config.time_step

    def warmup_duration(self, seq: PoseSequence) -> float:
        return self.warmup_frames / float(seq.fps)

    def load_target(self, plasticity_dir: str | os.PathLike[str]) -> None:
        plasticity_dir = Path(plasticity_dir)
        missing = []
        targets = []
        for seq_name in self.expected_sequence_names():
            path = plasticity_dir / f"{seq_name}.npy"
            if not path.is_file():
                missing.append(seq_name)
                continue
            targets.append(self._load_target_file(path))

        if missing:
            shown = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
            raise FileNotFoundError(
                f"Missing plasticity .npy files in {plasticity_dir}: {shown}{more}"
            )
        self.targets = targets
        self.is_zero_target = False

    def reflect(self, poses: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = poses[..., self.reflection_map, :]
        poses *= np.float32([1, 1, -1, -1])
        trans *= np.float32([-1, 1, 1])
        return poses, trans

    def __len__(self) -> int:
        if self.mode == "test":
            return len(self.sequences)
        return ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int):
        if self.mode == "test":
            return self._get_test_item(idx)
        return self._get_training_batch(idx)

    def on_epoch_end(self) -> None:
        if self.mode == "train":
            np.random.shuffle(self.sampled_id)

    def _get_test_item(self, idx: int):
        seq = self.sequences[idx]
        query_times = self._full_sequence_query_times(seq)
        poses, trans = self._get_pose_window(seq, query_times)
        phase = self._make_phase_window(seq, query_times)
        frames = np.array([seq.num_frames])
        output_steps = len(query_times) - 2
        plasticity_shape = (output_steps, self.edge_count, 1)
        plasticity_target = np.zeros(plasticity_shape, np.float32)
        plasticity_input = np.zeros(plasticity_shape, np.float32)

        if self.targets is not None and not self.is_zero_target:
            self._fill_target(plasticity_target, self.targets[idx])
        elif self.target_noise_as_target:
            self._add_supervised_target_noise(plasticity_target)
        plasticity_input[...] = plasticity_target
        if not self.target_noise_as_target:
            self._add_input_noise(plasticity_input, phase)

        x = {"poses": poses, "trans": trans, "phase": phase}
        y = {"target": plasticity_target, "target_input": plasticity_input, "frames": frames}
        return x, y

    def _full_sequence_query_times(self, seq: PoseSequence) -> np.ndarray:
        start_frame = -self.warmup_frames - 2
        end_frame = seq.num_frames - 1
        frames = np.arange(start_frame, end_frame + 1, dtype=np.float32)
        return frames * self.config.time_step

    def _get_training_batch(self, idx: int):
        start = idx * self.batch_size
        end = start + self.batch_size
        samples = self.samples[self.sampled_id[start:end]]
        batch_size = len(samples)

        poses = np.zeros((batch_size, *self.sample_poses_shape), np.float32)
        trans = np.zeros((batch_size, *self.sample_trans_shape), np.float32)
        phase = np.zeros((batch_size, *self.sample_phase_shape), np.float32)
        frames = np.zeros((batch_size, 1), np.float32)
        plasticity_target = np.zeros((batch_size, *self.sample_plasticity_shape), np.float32)
        plasticity_input = np.zeros((batch_size, *self.sample_plasticity_shape), np.float32)

        for i, (seq_idx, time, frame_count) in enumerate(samples):
            seq_idx = int(seq_idx)
            query_times = self.time_steps(time)
            seq = self.sequences[seq_idx]
            poses[i], trans[i] = self._get_pose_window(seq, query_times)
            phase[i] = self._make_phase_window(seq, query_times)
            frames[i] = frame_count
            if self.mode == "train" and np.random.uniform() < self.reflect_probability:
                poses[i], trans[i] = self.reflect(poses[i], trans[i])
            if self.targets is not None and not self.is_zero_target:
                self._fill_target(plasticity_target[i], self.targets[seq_idx])
                if self._use_zero_rest_rehearsal():
                    plasticity_target[i] = 0.0
            elif self.target_noise_as_target:
                self._add_supervised_target_noise(
                    plasticity_target[i],
                    key=(seq_idx, self._time_key(time)),
                )
            plasticity_input[i] = plasticity_target[i]
            if not self.target_noise_as_target:
                self._add_input_noise(plasticity_input[i], phase[i])

        x = {"poses": poses, "trans": trans, "phase": phase}
        y = {"target": plasticity_target, "target_input": plasticity_input, "frames": frames}
        return x, y

    def _use_zero_rest_rehearsal(self) -> bool:
        return (
            self.mode == "train"
            and self.targets is not None
            and not self.is_zero_target
            and self.rest_zero_rehearsal_probability > 0.0
            and np.random.uniform() < self.rest_zero_rehearsal_probability
        )

    def _read_sequence_names(
        self,
        max_sequences: int | None,
        sequence_names: list[str] | None,
    ) -> list[str]:
        with self.txt_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        names.sort()
        if sequence_names:
            names = [f"{Path(name).stem}.npz" for name in sequence_names]
            missing = [
                name
                for name in names
                if not (Path(DATA_DIR) / self.config.data.dataset / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing sequence files for {self.mode}: {missing}"
                )
        if max_sequences is not None:
            names = names[: int(max_sequences)]
        return names

    def _read_skeleton(self) -> None:
        fname = Path(BODY_DIR) / f"{self.config.body.skeleton}_skeleton.json"
        with fname.open("r", encoding="utf-8") as f:
            skel_data = json.load(f)
        self.joint_names = skel_data["joint_names"]
        self.num_joints = len(self.joint_names)

    def _make_reflection_map(self) -> None:
        self.reflection_map = [None] * len(self.joint_names)
        for name, idx in self.joint_names.items():
            reflected = name
            if "L" in name:
                reflected = name.replace("L", "R")
            elif "R" in name:
                reflected = name.replace("R", "L")
            self.reflection_map[idx] = self.joint_names[reflected]

    def _make_sample_list(self) -> None:
        samples = []
        for idx in self.seq_idx:
            if self.mode == "train":
                start_time = -self.warmup_duration(self.sequences[idx])
                times = np.arange(
                    start_time,
                    self.seq_duration[idx] + np.finfo(np.float32).eps,
                    self.config.time_step,
                )
            else:
                times = np.array([self.seq_duration[idx]])
            frames = np.tile(self.seq_frames[idx], len(times))
            samples += list(zip([idx] * len(times), times, frames))
        self.samples = np.array(samples)
        self.sampled_id = np.arange(len(self.samples))
        self.on_epoch_end()

    def _get_pose_window(
        self,
        seq: PoseSequence,
        query_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.warmup_frames <= 0:
            return seq.get(query_times, extrapolation="clip")

        poses, trans = seq.get(np.maximum(query_times, 0.0), extrapolation="clip")
        warmup = query_times < 0.0
        if not np.any(warmup):
            return poses, trans

        duration = max(self.warmup_duration(seq), np.finfo(np.float32).eps)
        alpha = np.clip((query_times[warmup] + duration) / duration, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        t_pose = np.zeros((len(alpha), *seq.skeleton_shape), np.float32)
        t_pose[..., 0] = 1.0
        first_pose = np.broadcast_to(seq.poses[0], t_pose.shape).astype(np.float32)
        from utils.rotation import slerp

        poses[warmup] = slerp(t_pose, first_pose, alpha).astype(np.float32)
        trans[warmup] = seq.trans[0]
        return poses, trans

    def _make_phase_window(self, seq: PoseSequence, query_times: np.ndarray) -> np.ndarray:
        effective_times = query_times[2:]
        phase = np.zeros((len(effective_times), 2), np.float32)
        duration = max(float(seq.duration), np.finfo(np.float32).eps)
        phase[:, 0] = np.clip(effective_times / duration, 0.0, 1.0)
        phase[:, 1] = (effective_times < 0.0).astype(np.float32)
        return phase

    def _fill_target(self, out: np.ndarray, target: np.ndarray) -> None:
        pad_len = out.shape[0] - target.shape[0]
        if pad_len < 0:
            target = target[-out.shape[0] :]
            pad_len = 0
        out[:pad_len, :, :] = target[0:1, :, None]
        out[pad_len:, :, :] = target[..., None]

    def _load_target_file(self, path: Path) -> np.ndarray:
        target = np.load(path).astype(np.float32)
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target[..., 0]
        if target.ndim != 2:
            raise ValueError(f"{path} must have shape [T, E] or [T, E, 1], got {target.shape}")
        if target.shape[1] != self.edge_count:
            raise ValueError(
                f"{path} edge count mismatch: expected {self.edge_count}, got {target.shape[1]}"
            )
        if target.shape[0] == 0:
            raise ValueError(f"{path} is empty along the time dimension")
        return target

    def _add_input_noise(self, target_input: np.ndarray, phase: np.ndarray) -> None:
        if self.target_input_noise_std <= 0:
            return
        noise = np.random.normal(
            loc=0.0,
            scale=self.target_input_noise_std,
            size=target_input.shape,
        ).astype(np.float32)
        target_input += noise * (1.0 - phase[:, 1:2, None])

    def _time_key(self, time: float) -> int:
        return int(round(float(time) / float(self.config.time_step)))

    def _add_supervised_target_noise(
        self,
        target: np.ndarray,
        key: tuple[int, int] | None = None,
    ) -> None:
        if self.target_input_noise_std <= 0:
            return
        if key is not None and key in self._initial_target_noise_cache:
            target += self._initial_target_noise_cache[key]
            return

        edge_noise = np.random.normal(
            loc=0.0,
            scale=self.target_input_noise_std,
            size=(target.shape[1], target.shape[2]),
        ).astype(np.float32)
        edge_noise = self._smooth_edge_noise(edge_noise)
        edge_noise = self._mask_edge_noise(edge_noise)
        target_noise = np.broadcast_to(edge_noise[None], target.shape).copy()
        if key is not None:
            self._initial_target_noise_cache[key] = target_noise
        target += target_noise

    def _make_edge_neighbors(
        self,
        edge_neighbors: list[np.ndarray] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if edge_neighbors is None:
            return None, None
        neighbors = []
        for edge_neighbors_i in edge_neighbors[: self.edge_count]:
            neighbors_i = np.asarray(edge_neighbors_i, dtype=np.int32)
            neighbors_i = neighbors_i[neighbors_i < self.edge_count]
            neighbors.append(neighbors_i)

        max_count = max((len(neighbors_i) for neighbors_i in neighbors), default=0)
        if max_count == 0:
            return None, None
        indices = np.zeros((self.edge_count, max_count), dtype=np.int32)
        mask = np.zeros((self.edge_count, max_count, 1), dtype=np.float32)
        for edge_idx, neighbors_i in enumerate(neighbors):
            count = len(neighbors_i)
            if count == 0:
                indices[edge_idx] = edge_idx
                continue
            indices[edge_idx, :count] = neighbors_i
            mask[edge_idx, :count, 0] = 1.0
        return indices, mask

    def _make_target_noise_edge_mask(
        self,
        target_noise_edge_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if target_noise_edge_mask is None:
            return None
        mask = np.asarray(target_noise_edge_mask[: self.edge_count], dtype=np.float32)
        return mask.reshape(self.edge_count, 1)

    def _mask_edge_noise(self, noise: np.ndarray) -> np.ndarray:
        if self.target_noise_edge_mask is None:
            return noise
        masked = noise * self.target_noise_edge_mask
        current_std = float(np.std(masked[self.target_noise_edge_mask[:, 0] > 0]))
        if current_std > 0:
            masked *= self.target_input_noise_std / current_std
        return masked.astype(np.float32)

    def _smooth_edge_noise(self, noise: np.ndarray) -> np.ndarray:
        if (
            self.edge_neighbor_indices is None
            or self.edge_neighbor_mask is None
            or self.target_noise_smoothing_steps <= 0
        ):
            return noise
        smoothed = noise
        for _ in range(self.target_noise_smoothing_steps):
            gathered = smoothed[self.edge_neighbor_indices]
            neighbor_sum = np.sum(gathered * self.edge_neighbor_mask, axis=1)
            neighbor_count = np.maximum(np.sum(self.edge_neighbor_mask, axis=1), 1.0)
            neighbor_mean = neighbor_sum / neighbor_count
            smoothed = 0.5 * smoothed + 0.5 * neighbor_mean
        current_std = float(np.std(smoothed))
        if current_std > 0:
            smoothed *= self.target_input_noise_std / current_std
        return smoothed.astype(np.float32)
