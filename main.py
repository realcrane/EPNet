from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

import global_vars as gv
from global_vars import BODY_DIR, CHECKPOINTS_DIR, ELASTICITY_DIR, LOGS_DIR, RENDER_DIR
from model.build import build_ncs_model
from model.cloth import Garment
from epnet.data import ElasticityDataset
from utils import debug
from utils.collision_projection import project_cloth_outside_body
from utils.IO import writePC2Frames
from utils.config import MainConfig

mixed_precision.set_global_policy("mixed_float16")


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


def make_output_dirs(results_dir: Path | None) -> dict[str, Path]:
    if results_dir is None:
        return {
            "checkpoints": Path(CHECKPOINTS_DIR),
            "elasticity": Path(ELASTICITY_DIR),
            "logs": Path(LOGS_DIR),
            "render": Path(RENDER_DIR),
        }
    return {
        "checkpoints": results_dir / "checkpoints",
        "elasticity": results_dir / "elasticity",
        "logs": results_dir / "logs",
        "render": results_dir / "render",
    }


def save_elastic_threshold(
    elasticity_model,
    data: ElasticityDataset,
    config: MainConfig,
    iteration: int,
    output_dirs: dict[str, Path],
) -> None:
    elasticity_path = output_dirs["elasticity"] / f"iteration_{iteration}"
    garment_path = elasticity_path / "garment"
    threshold_path = elasticity_path / "threshold"
    garment_path.mkdir(parents=True, exist_ok=True)
    threshold_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(data):
        body, garment, unskinned, plasticity_target, frames = elasticity_model.predict(batch, w=1.0)
        name = Path(data.files_name[i]).stem
        frame_count = int(frames[0])

        print("File name: [", name, "] Frame count: ", frame_count)
        angle = debug.deformation_to_signed_angle(garment, elasticity_model.garment)
        angle_delta = angle - elasticity_model.garment.face_dir_dihedral
        angle_delta = tf.math.atan2(tf.sin(angle_delta), tf.cos(angle_delta))
        angle_delta = angle_delta[:, -frame_count:]
        threshold = debug.signed_angle_to_threshold(
            angle_delta[..., None],
            config.step_smooth_scale,
            config.angle_step,
        )

        render_path = output_dirs["render"] / f"iteration_{iteration}" / name
        render_path.mkdir(parents=True, exist_ok=True)

        body = np.array(body[0])[-frame_count:]
        garment = np.array(garment[0])[-frame_count:]
        unskinned = np.array(unskinned[0])[-frame_count:]
        threshold = np.array(threshold[0, :, :, 0])[-frame_count:]
        collision_projection_threshold = float(
            getattr(config.loss, "collision_projection_threshold", 0.0)
        )
        garment_render = project_cloth_outside_body(
            body,
            garment,
            elasticity_model.body.faces,
            collision_projection_threshold,
        )

        np.save(garment_path / f"{name}.npy", garment)
        np.save(threshold_path / f"{name}.npy", threshold)

        pc2_overwrite(render_path / "body.pc2", body)
        pc2_overwrite(render_path / "tshirt.pc2", garment_render)
        pc2_overwrite(render_path / "tshirt_unskinned.pc2", unskinned)


class EpochPreviewCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model_ref,
        data: ElasticityDataset,
        config: MainConfig,
        iteration: int,
        output_dirs: dict[str, Path],
        preview_epochs: set[int],
    ) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.data = data
        self.config = config
        self.iteration = iteration
        self.output_dirs = output_dirs
        self.preview_epochs = preview_epochs

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        epoch_num = epoch + 1
        if epoch_num not in self.preview_epochs:
            return
        print(f"Save epoch preview: iteration {self.iteration}, epoch {epoch_num}")
        preview_dirs = dict(self.output_dirs)
        preview_dirs["render"] = self.output_dirs["render"] / "epoch_preview"
        save_elastic_threshold(
            self.model_ref,
            self.data,
            self.config,
            f"{self.iteration}_epoch_{epoch_num}",
            preview_dirs,
        )


def load_previous_thresholds(
    train_data: ElasticityDataset,
    validation_data: ElasticityDataset,
    test_data: ElasticityDataset,
    iteration: int,
    output_dirs: dict[str, Path],
) -> None:
    threshold_path = output_dirs["elasticity"] / f"iteration_{iteration - 1}" / "threshold"
    train_data.load_target(threshold_path)
    validation_data.load_target(threshold_path)
    test_data.load_target(threshold_path)


def make_target_noise_edge_mask(garment: Garment) -> np.ndarray:
    mask = np.ones(len(garment.edge_adjacency_index), dtype=np.float32)
    if not garment.pinning:
        return mask
    pinned = np.zeros(garment.num_verts, dtype=bool)
    pinned[garment.pinning_vertices] = True
    edge_vertices = garment.face_adjacency_edges[: len(mask)]
    mask[np.any(pinned[edge_vertices], axis=1)] = 0.0
    return mask


def elastic_epochs_for_iteration(
    iteration: int,
    config: MainConfig,
    elasticity_epochs: int | None,
    elasticity_initial_epochs: int | None,
    elasticity_later_epochs: int | None,
) -> int:
    if iteration == 0 and elasticity_initial_epochs is not None:
        return int(elasticity_initial_epochs)
    if iteration > 0 and elasticity_later_epochs is not None:
        return int(elasticity_later_epochs)
    if elasticity_epochs is not None:
        return int(elasticity_epochs)
    return int(config.experiment.elasticity_epochs)


def main(
    config: MainConfig,
    start_iteration: int | None = None,
    end_iteration: int | None = None,
    target_dir: Path | None = None,
    sequence_names: list[str] | None = None,
    results_dir: Path | None = None,
    elasticity_epochs: int | None = None,
    elasticity_initial_epochs: int | None = None,
    elasticity_later_epochs: int | None = None,
    target_input_noise_std: float = 0.0,
    target_noise_as_target: bool = False,
    target_noise_smoothing_steps: int = 0,
    warmup_frames: int = 0,
    enable_tensorboard: bool = False,
    checkpoint_each_epoch: bool = False,
    preview_epochs: list[int] | None = None,
    rest_zero_rehearsal_probability: float = 0.0,
) -> None:
    output_dirs = make_output_dirs(results_dir)
    print("Preparing elastic training data...")
    garment_obj = Path(BODY_DIR) / config.body.model / config.garment.name
    garment = Garment(str(garment_obj))
    edge_count = len(garment.edge_adjacency_index)
    target_noise_edge_mask = make_target_noise_edge_mask(garment)

    print("Building elastic network...")
    elasticity_model = build_ncs_model(config, edge_count)

    train_data = ElasticityDataset(
        config,
        edge_count,
        mode="train",
        sequence_names=sequence_names,
        target_noise_smoothing_steps=target_noise_smoothing_steps,
        warmup_frames=warmup_frames,
        edge_neighbors=garment.edges_neighbours,
        target_noise_edge_mask=target_noise_edge_mask,
        rest_zero_rehearsal_probability=rest_zero_rehearsal_probability,
    )
    validation_data = ElasticityDataset(
        config,
        edge_count,
        mode="validation",
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
        edge_neighbors=garment.edges_neighbours,
    )
    test_data = ElasticityDataset(
        config,
        edge_count,
        mode="test",
        sequence_names=sequence_names,
        warmup_frames=warmup_frames,
        edge_neighbors=garment.edges_neighbours,
    )

    start = 0
    if start_iteration is not None:
        start = int(start_iteration)
        if start > 0:
            checkpoint_path = (
                output_dirs["checkpoints"]
                / config.experiment.elasticity_checkpoint
                / f"{config.name}_{start - 1}"
            )
            print("Load checkpoint iteration [", start - 1, "]")
            elasticity_model.load_weights(str(checkpoint_path))
    elif config.experiment.elasticity_load_iter is not None:
        start = int(config.experiment.elasticity_load_iter)
        checkpoint_path = (
            output_dirs["checkpoints"]
            / config.experiment.elasticity_checkpoint
            / f"{config.name}_{start}"
        )
        print("Load checkpoint iteration [", start, "]")
        elasticity_model.load_weights(str(checkpoint_path))
        start += 1

    stop = config.training_iteration if end_iteration is None else int(end_iteration)
    for iteration in range(start, stop):
        print("Current iteration: ", iteration)
        train_data.target_input_noise_std = (
            float(target_input_noise_std) if iteration == 0 else 0.0
        )
        train_data.target_noise_as_target = bool(target_noise_as_target and iteration == 0)
        validation_data.target_input_noise_std = 0.0
        validation_data.target_noise_as_target = False
        test_data.target_input_noise_std = 0.0
        test_data.target_noise_as_target = False

        if target_dir is not None:
            train_data.target_noise_as_target = False
            train_data.load_target(target_dir)
            validation_data.load_target(target_dir)
            test_data.load_target(target_dir)
        elif iteration > 0:
            train_data.target_noise_as_target = False
            load_previous_thresholds(train_data, validation_data, test_data, iteration, output_dirs)

        callbacks = []
        if enable_tensorboard:
            gv.tensorboard_callback_elasticity = tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dirs["logs"] / f"ncs_{config.name}_{iteration}"),
                histogram_freq=0,
            )
            callbacks.append(gv.tensorboard_callback_elasticity)

        checkpoint_path = output_dirs["checkpoints"] / "ncs" / f"{config.name}_{iteration}"
        if checkpoint_each_epoch:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    save_freq="epoch",
                )
            )
        active_preview_epochs = {
            int(epoch)
            for epoch in (preview_epochs or [])
            if int(epoch) > 0
        }
        if active_preview_epochs:
            callbacks.append(
                EpochPreviewCallback(
                    elasticity_model,
                    test_data,
                    config,
                    iteration,
                    output_dirs,
                    active_preview_epochs,
                )
            )

        print("Elasticity model training...")
        current_epochs = elastic_epochs_for_iteration(
            iteration,
            config,
            elasticity_epochs,
            elasticity_initial_epochs,
            elasticity_later_epochs,
        )
        print("Elasticity epochs: ", current_epochs)
        elasticity_model.fit(
            train_data,
            validation_data=validation_data,
            epochs=current_epochs,
            callbacks=callbacks,
        )

        if not checkpoint_each_epoch:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            elasticity_model.save(str(checkpoint_path))

        print("Save elasticity deformation and threshold")
        save_elastic_threshold(elasticity_model, test_data, config, iteration, output_dirs)

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EPNet elastic/NCS model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", required=True)
    parser.add_argument("--start-iteration", type=int, default=None)
    parser.add_argument("--end-iteration", type=int, default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--sequence-name", action="append", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--elasticity-epochs", type=int, default=None)
    parser.add_argument("--elasticity-initial-epochs", type=int, default=None)
    parser.add_argument("--elasticity-later-epochs", type=int, default=None)
    parser.add_argument("--target-input-noise-std", type=float, default=0.0)
    parser.add_argument("--target-noise-as-target", action="store_true")
    parser.add_argument("--target-noise-smoothing-steps", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--checkpoint-each-epoch", action="store_true")
    parser.add_argument("--preview-epoch", type=int, action="append", default=None)
    parser.add_argument("--rest-zero-rehearsal-probability", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_gpu(args.gpu_id)
    main(
        MainConfig(args.config),
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
        target_dir=Path(args.target_dir) if args.target_dir else None,
        sequence_names=args.sequence_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        elasticity_epochs=args.elasticity_epochs,
        elasticity_initial_epochs=args.elasticity_initial_epochs,
        elasticity_later_epochs=args.elasticity_later_epochs,
        target_input_noise_std=args.target_input_noise_std,
        target_noise_as_target=args.target_noise_as_target,
        target_noise_smoothing_steps=args.target_noise_smoothing_steps,
        warmup_frames=args.warmup_frames,
        enable_tensorboard=args.tensorboard,
        checkpoint_each_epoch=args.checkpoint_each_epoch,
        preview_epochs=args.preview_epoch,
        rest_zero_rehearsal_probability=args.rest_zero_rehearsal_probability,
    )
