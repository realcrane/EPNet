from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU

from global_vars import BODY_DIR
from loss.losses import (
    BendingLoss,
    ClothLoss,
    CollisionLoss,
    EdgeLoss,
    GravityLoss,
    InertiaLoss,
    PinningLoss,
    StVKLoss,
)
from loss.metrics import MyMetric
from model.body import Body
from model.cloth import Garment
from model.layers import Collision, FullyConnected, LBS, PSD, Rotation, Skeleton, SkelFlatten
from utils.tensor import compute_nth_derivative


def tf_shape(tensor):
    return [(size or -1) for size in tensor.get_shape()]


def config_value(config, path, default):
    value = config
    for key in path.split("."):
        if not hasattr(value, key):
            return default
        value = getattr(value, key)
    return value


class NCS(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        asset_dir = os.path.join(BODY_DIR, config.body.model)
        body_model = os.path.join(asset_dir, "body.npz")
        garment_obj = os.path.join(asset_dir, config.garment.name)

        print("Reading body model...")
        self.body = Body(body_model, input_joints=config.body.input_joints)

        print("Reading garment...")
        self.garment = Garment(garment_obj)

        print("Computing cloth blend weights...")
        self.garment.transfer_blend_weights(self.body)

        print("Smoothing cloth blend weights...")
        self.garment.smooth_blend_weights(
            iterations=config.garment.blend_weights_smoothing_iterations
        )

        self.build_model()
        self.build_losses_and_metrics()

    def build_model(self):
        self.build_lbs()
        self.build_encoder()
        self.build_decoder()

    def build_lbs(self):
        self.rot = Rotation(name="Rotation")
        self.skeleton = Skeleton(self.body.joints, name="Skeleton")
        self.lbs_body = LBS(self.body.blend_weights, trainable=False, name="LBS_Body")
        self.lbs_cloth = LBS(
            self.garment.blend_weights,
            trainable=self.config.blend_weights_trainable,
            name="LBS_Cloth",
        )

    def build_encoder(self):
        self.plasticity_encoder = [
            SkelFlatten(),
            FullyConnected(256, act=tf.nn.relu, name="plt_enc_fc0"),
            FullyConnected(128, act=tf.nn.relu, name="plt_enc_fc1"),
            FullyConnected(64, act=tf.nn.relu, name="plt_enc_fc2"),
        ]

        self.static_encoder = [
            SkelFlatten(),
            FullyConnected(64, act=tf.nn.relu, name="stc_enc_fc0"),
            FullyConnected(128, act=tf.nn.relu, name="stc_enc_fc1"),
            FullyConnected(256, act=tf.nn.relu, name="stc_enc_fc2"),
            FullyConnected(512, act=tf.nn.relu, name="stc_enc_fc3"),
        ]

        self.dynamic_encoder = [
            FullyConnected(32, act=tf.nn.relu, use_bias=False, name="dyn_enc_fc0"),
            FullyConnected(32, act=tf.nn.relu, use_bias=False, name="dyn_enc_fc1"),
            SkelFlatten(),
            FullyConnected(512, act=tf.nn.relu, use_bias=False, name="dyn_enc_fc2"),
            FullyConnected(512, act=tf.nn.relu, use_bias=False, name="dyn_enc_fc3"),
            GRU(512, use_bias=False, return_sequences=True, name="dyn_enc_gru"),
        ]

        self.phase_encoder = [
            FullyConnected(16, act=tf.nn.relu, name="phase_enc_fc0"),
            FullyConnected(32, act=tf.nn.relu, name="phase_enc_fc1"),
        ]

    def build_decoder(self):
        self.decoder = [
            FullyConnected(512, act=tf.nn.relu, name="dec_fc0"),
            FullyConnected(512, act=tf.nn.relu, name="dec_fc1"),
            FullyConnected(512, act=tf.nn.relu, name="dec_fc2"),
        ]
        self.PSD = PSD(self.garment.num_verts, name="dec_PSD")

    def build_losses_and_metrics(self):
        self.loss_metric = MyMetric(name="Loss")

        if self.config.cloth_model == "mass-spring":
            self.cloth_loss = EdgeLoss(self.garment)
            self.edge_metric = MyMetric(name="Edge")
        elif self.config.cloth_model == "baraff98":
            self.cloth_loss = ClothLoss(self.garment)
            self.stretch_metric = MyMetric(name="Stretch")
            self.shear_metric = MyMetric(name="Shear")
        elif self.config.cloth_model == "stvk":
            self.cloth_loss = StVKLoss(
                self.garment,
                self.config.loss.cloth.lambda_,
                self.config.loss.cloth.mu,
            )
            self.strain_metric = MyMetric(name="Strain")
        else:
            raise ValueError(f"Unsupported cloth model: {self.config.cloth_model}")

        self.bending_loss = BendingLoss(self.garment)
        self.bending_metric = MyMetric(name="Bending")
        self.rest_conditioning_metric = MyMetric(name="RestConditioning")
        self.zero_rest_metric = MyMetric(name="ZeroRest")
        self.rest_response_metric = MyMetric(name="RestResponse")

        self.collision = Collision(self.body, use_ray=False, name="Collision")
        self.collision_loss = CollisionLoss(
            self.body,
            collision_threshold=self.config.loss.collision_threshold,
            vertex_weights=self.local_collision_vertex_weights(),
        )
        self.collision_metric = MyMetric(name="Collision")

        self.gravity_loss = GravityLoss(
            self.garment.vertex_area,
            density=self.config.loss.density,
            gravity=self.config.gravity,
        )
        self.gravity_metric = MyMetric(name="Gravity")

        self.inertia_loss = InertiaLoss(
            self.config.time_step,
            self.garment.vertex_area,
            density=self.config.loss.density,
        )
        self.inertia_metric = MyMetric(name="Inertia")

        if self.garment.pinning:
            self.pinning_loss = PinningLoss(self.garment, self.config.pin_blend_weights)

    def local_collision_vertex_weights(self):
        weight = float(config_value(self.config.loss, "local_collision.weight", 0.0))
        if weight <= 0.0:
            return None

        vertices = self.garment.vertices
        x_abs_min = float(config_value(self.config.loss, "local_collision.x_abs_min", 0.22))
        x_abs_max = float(config_value(self.config.loss, "local_collision.x_abs_max", 0.32))
        y_min = float(config_value(self.config.loss, "local_collision.y_min", 0.33))
        y_max = float(config_value(self.config.loss, "local_collision.y_max", 0.39))
        z_min = float(config_value(self.config.loss, "local_collision.z_min", -0.08))
        z_max = float(config_value(self.config.loss, "local_collision.z_max", -0.015))
        mask = (
            (np.abs(vertices[:, 0]) >= x_abs_min)
            & (np.abs(vertices[:, 0]) <= x_abs_max)
            & (vertices[:, 1] >= y_min)
            & (vertices[:, 1] <= y_max)
            & (vertices[:, 2] >= z_min)
            & (vertices[:, 2] <= z_max)
        )

        weights = np.ones((vertices.shape[0],), dtype=np.float32)
        weights[mask] += weight
        print(f"Local collision vertices: {int(mask.sum())} weight={weight}")
        return weights

    @property
    def metrics(self):
        if self.config.cloth_model == "mass-spring":
            cloth_metrics = [self.edge_metric]
        elif self.config.cloth_model == "baraff98":
            cloth_metrics = [self.stretch_metric, self.shear_metric]
        elif self.config.cloth_model == "stvk":
            cloth_metrics = [self.strain_metric]
        else:
            cloth_metrics = []

        return [
            self.loss_metric,
            *cloth_metrics,
            self.bending_metric,
            self.rest_conditioning_metric,
            self.zero_rest_metric,
            self.rest_response_metric,
            self.collision_metric,
            self.gravity_metric,
            self.inertia_metric,
        ]

    @property
    def cloth_blend_weights(self):
        return self.lbs_cloth.blend_weights

    def compute_losses_and_metrics(
        self,
        body,
        vertices,
        unskinned,
        plasticity_target,
        training,
    ):
        loss = self.compute_static_losses_and_metrics(
            body,
            vertices[:, -1],
            unskinned,
            plasticity_target[:, -1],
        )

        if training and self.config.motion_augmentation:
            vertices = vertices[self.config.motion_augmentation :]
        return loss + self.compute_dynamic_losses_and_metrics(vertices)

    def compute_static_losses_and_metrics(self, body, vertices, unskinned, plasticity_target):
        if self.config.cloth_model == "mass-spring":
            cloth_loss, edge_error = self.cloth_loss(vertices)
            cloth_loss *= self.config.loss.cloth.edge
        elif self.config.cloth_model == "baraff98":
            stretch_loss, stretch_error, shear_loss, shear_error = self.cloth_loss(vertices)
            cloth_loss = (
                self.config.loss.cloth.stretch * stretch_loss
                + self.config.loss.cloth.shear * shear_loss
            )
        elif self.config.cloth_model == "stvk":
            cloth_loss, strain_error = self.cloth_loss(vertices)

        bending_loss, bending_error = self.bending_loss(vertices, plasticity_target)
        rest_conditioning_loss, rest_conditioning_error = self.rest_conditioning_loss(
            vertices,
            plasticity_target,
        )

        collision_indices = self.collision(vertices, body)
        collision_loss, collision_error = self.collision_loss(
            vertices,
            body,
            collision_indices,
        )

        gravity_loss = self.gravity_loss(vertices)
        loss = (
            cloth_loss
            + self.config.loss.bending * bending_loss
            + rest_conditioning_loss
            + self.config.loss.collision_weight * collision_loss
            + gravity_loss
        )

        if self.garment.pinning:
            pinning_loss = self.pinning_loss(unskinned)
            loss += self.config.loss.pinning * pinning_loss

        self.loss_metric.update_state(loss)
        if self.config.cloth_model == "mass-spring":
            self.edge_metric.update_state(edge_error)
        elif self.config.cloth_model == "baraff98":
            self.stretch_metric.update_state(stretch_error)
            self.shear_metric.update_state(shear_error)
        elif self.config.cloth_model == "stvk":
            self.strain_metric.update_state(strain_error)
        self.bending_metric.update_state(bending_error)
        self.rest_conditioning_metric.update_state(rest_conditioning_error)
        self.collision_metric.update_state(collision_error)
        self.gravity_metric.update_state(gravity_loss)

        return loss

    def rest_conditioning_loss(self, vertices, plasticity_target):
        weight = float(config_value(self.config.loss, "rest_conditioning.weight", 0.0))
        if weight <= 0.0:
            return 0.0, 0.0

        zero_weight = float(config_value(self.config.loss, "rest_conditioning.zero_weight", 1.0))
        active_weight = float(config_value(self.config.loss, "rest_conditioning.active_weight", 1.0))
        threshold = float(config_value(self.config.loss, "rest_conditioning.active_threshold", 1e-4))

        target = tf.squeeze(tf.cast(plasticity_target, tf.float32))
        diff = self.bending_loss.signed_bending_diff(vertices, plasticity_target)
        active = tf.cast(tf.abs(target) > threshold, tf.float32)
        weights = zero_weight * (1.0 - active) + active_weight * active
        loss = weight * tf.reduce_mean(weights * tf.square(diff))
        error = tf.reduce_mean(tf.abs(diff))
        return loss, error

    def compute_dynamic_losses_and_metrics(self, vertices):
        inertia_loss, inertia_error = self.inertia_loss(vertices)
        self.inertia_metric.update_state(inertia_error)
        return inertia_loss

    def train_step(self, inputs):
        x, y = inputs
        plasticity_input = y.get("target_input", y["target"])
        phase = x.get("phase")
        with tf.GradientTape() as tape:
            body, vertices, unskinned, _ = self(
                (x["poses"], x["trans"], plasticity_input, phase),
                training=True,
            )
            loss = self.compute_losses_and_metrics(
                body,
                vertices,
                unskinned,
                y["target"],
                training=True,
            )
            loss += self.paired_rest_consistency_loss(
                x["poses"],
                x["trans"],
                plasticity_input,
                y["target"],
                phase,
                vertices,
                training=True,
            )
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def paired_rest_consistency_loss(
        self,
        poses,
        trans,
        plasticity_input,
        plasticity_target,
        phase,
        vertices,
        training,
    ):
        zero_weight = float(config_value(self.config.loss, "zero_rest_consistency.weight", 0.0))
        response_weight = float(config_value(self.config.loss, "rest_response.weight", 0.0))
        if zero_weight <= 0.0 and response_weight <= 0.0:
            return 0.0

        zero_plasticity = tf.zeros_like(plasticity_input)
        _, zero_vertices, _, _ = self(
            (poses, trans, zero_plasticity, phase),
            training=training,
            w=None,
        )
        zero_target = tf.zeros_like(zero_plasticity[:, -1])
        zero_angle = self.bending_loss.signed_bending_diff(zero_vertices[:, -1], zero_target)
        loss = zero_weight * tf.reduce_mean(tf.square(zero_angle))
        self.zero_rest_metric.update_state(tf.reduce_mean(tf.abs(zero_angle)))

        if response_weight > 0.0:
            target = tf.squeeze(tf.cast(plasticity_target[:, -1], tf.float32))
            normal_angle = self.bending_loss.signed_bending_diff(vertices[:, -1], zero_target)
            response_error = (normal_angle - zero_angle) - target
            loss += response_weight * tf.reduce_mean(tf.square(response_error))
            self.rest_response_metric.update_state(tf.reduce_mean(tf.abs(response_error)))

        return loss

    def test_step(self, inputs):
        x, y = inputs
        plasticity_input = y.get("target_input", y["target"])
        phase = x.get("phase")
        body, vertices, unskinned, _ = self(
            (x["poses"], x["trans"], plasticity_input, phase),
            training=False,
            predict=True,
        )
        self.compute_losses_and_metrics(
            body,
            vertices,
            unskinned,
            y["target"],
            training=False,
        )
        self.paired_rest_consistency_loss(
            x["poses"],
            x["trans"],
            plasticity_input,
            y["target"],
            phase,
            vertices,
            training=False,
        )
        return {m.name: m.result() for m in self.metrics}

    def predict(self, inputs, w):
        x, y = inputs
        poses = x["poses"]
        trans = x["trans"]
        plasticity_target = y["target"]
        plasticity_input = y.get("target_input", plasticity_target)
        phase = x.get("phase")
        frames = y["frames"]

        if poses.ndim == 3:
            poses = poses[None]
            trans = trans[None]
            plasticity_target = plasticity_target[None]
            plasticity_input = plasticity_input[None]
            if phase is not None:
                phase = phase[None]
            frames = frames[None]

        motion_features, matrices = self.call_inputs(poses, trans)
        deformations = self.call_network(
            motion_features,
            plasticity_input,
            phase,
            w=w,
            training=False,
            predict=True,
        )

        body = self.lbs_body(self.body.vertices, matrices)
        unskinned = self.garment.vertices + deformations
        matrices = tf.gather(matrices, self.body.input_joints, axis=-3)
        garment = self.lbs_cloth(unskinned, matrices)

        return body, garment, unskinned, plasticity_target, frames

    def call(self, inputs, w=None, training=False, predict=False):
        poses, trans, plasticity_target, *optional = inputs
        phase = optional[0] if optional else None

        motion_features, matrices = self.call_inputs(poses, trans)
        deformations = self.call_network(
            motion_features,
            plasticity_target,
            phase,
            w=w,
            training=training,
        )

        body = self.lbs_body(self.body.vertices, matrices[:, -1])
        unskinned = self.garment.vertices + deformations
        cloth_matrices = tf.gather(matrices, self.body.input_joints, axis=-3)
        garment = self.lbs_cloth(unskinned, cloth_matrices[:, -3:])

        return body, garment, unskinned[:, -1], plasticity_target

    def call_inputs(self, poses, trans):
        rotations = self.rot(poses)
        matrices = self.body.forward_kinematics(rotations, trans)
        inverse_rotations = tf.linalg.matrix_transpose(matrices[..., :3])

        joint_pose = tf.reshape(rotations[..., :2], (*tf_shape(rotations)[:-2], 6))
        unposed_gravity = (
            inverse_rotations
            @ self.gravity_loss.gravity[:, None]
            * (1 / tf.norm(self.gravity_loss.gravity))
        )[..., 0]
        unposed_gravity = tf.cast(unposed_gravity, joint_pose.dtype)
        joint_pose = tf.concat((joint_pose, unposed_gravity), axis=-1)

        joint_positions = self.skeleton(matrices)
        pose_velocity = compute_nth_derivative(joint_pose, 1, self.config.time_step)[:, 1:]
        joint_acceleration = compute_nth_derivative(
            joint_positions,
            2,
            self.config.time_step,
        )
        joint_acceleration = (inverse_rotations[:, 2:] @ joint_acceleration[..., None])[..., 0]

        dtype = joint_pose.dtype
        pose_velocity = tf.cast(pose_velocity, dtype)
        joint_acceleration = tf.cast(joint_acceleration, dtype)
        motion_features = tf.concat(
            (joint_pose[:, 2:], pose_velocity, joint_acceleration),
            axis=-1,
        )
        motion_features = tf.gather(motion_features, self.body.input_joints, axis=-2)
        return motion_features, matrices[:, 2:]

    def call_network(self, features, plasticity_target, phase, w, training, predict=False):
        static_features, dynamic_features = tf.split(features, [9, 12], axis=-1)
        plasticity_features = plasticity_target
        if phase is None:
            phase = tf.zeros(
                tf.concat([tf.shape(plasticity_target)[:2], [2]], axis=0),
                dtype=plasticity_target.dtype,
            )
        phase = tf.cast(phase, static_features.dtype)

        if not predict:
            static_features = static_features[:, -3:]
            plasticity_features = plasticity_features[:, -3:]
            phase = phase[:, -3:]

        for layer in self.plasticity_encoder:
            plasticity_features = layer(plasticity_features)
        for layer in self.static_encoder:
            static_features = layer(static_features)
        for layer in self.dynamic_encoder:
            dynamic_features = layer(dynamic_features)
        for layer in self.phase_encoder:
            phase = layer(phase)

        if not predict:
            dynamic_features = dynamic_features[:, -3:]

        if training and self.config.motion_augmentation:
            static_features, dynamic_features = self.motion_augmentation(
                static_features,
                dynamic_features,
            )
        if w is not None:
            dynamic_features = w * dynamic_features

        decoded = static_features + dynamic_features
        decoded = tf.concat((decoded, plasticity_features, phase), axis=-1)
        for layer in self.decoder:
            decoded = layer(decoded)

        return self.PSD(decoded)

    def motion_augmentation(self, static_features, dynamic_features):
        batch_size = tf.shape(static_features)[0]
        num_augmented = self.config.motion_augmentation
        splits = [num_augmented, tf.maximum(0, batch_size - num_augmented)]

        static_aug, static_base = tf.split(static_features, splits)
        dynamic_aug, dynamic_base = tf.split(dynamic_features, splits)

        static_aug = tf.stop_gradient(static_aug)
        dynamic_aug = tf.stop_gradient(tf.random.shuffle(dynamic_aug))

        static_features = tf.concat((static_aug, static_base), axis=0)
        dynamic_features = tf.concat((dynamic_aug, dynamic_base), axis=0)
        return static_features, dynamic_features
