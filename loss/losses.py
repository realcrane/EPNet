import tensorflow as tf
from utils.mesh import vertex_normals, face_normals
import numpy as np
# Mass-spring model
class EdgeLoss:
    def __init__(self, garment):
        self.edges = garment.edges
        self.edge_lengths_true = garment.edge_lengths

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        edges = tf.gather(vertices, self.edges[:, 0], axis=1) - tf.gather(
            vertices, self.edges[:, 1], axis=1
        )
        edge_lengths = tf.norm(edges, axis=-1)
        print("<", edge_lengths, ">")
        edge_difference = edge_lengths - self.edge_lengths_true
        loss = edge_difference**2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.abs(edge_difference)
        error = tf.reduce_mean(error)
        return loss, error


# Baraff '98 cloth model (squared)
class ClothLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices
        pass

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=2,
        )
        w = tf.einsum("abcd,bce->abed", dX, self.uv_matrices)

        stretch = tf.norm(w, axis=-1) - 1

        stretch_loss = self.face_areas[:, None] * stretch**2

        stretch_loss = tf.reduce_sum(stretch_loss, axis=[1, 2])
        stretch_loss = tf.reduce_mean(stretch_loss)

        stretch_error = (
            self.face_areas[:, None] * tf.abs(stretch) * (0.5 / self.total_area)
        )
        stretch_error = tf.reduce_mean(tf.reduce_sum(stretch_error, axis=-1))

        shear = tf.reduce_sum(tf.multiply(w[:, :, 0], w[:, :, 1]), axis=-1)
        shear_loss = shear**2
        shear_loss *= self.face_areas
        shear_loss = tf.reduce_sum(shear_loss, axis=1)
        shear_loss = tf.reduce_mean(shear_loss)
        shear_error = self.face_areas * tf.abs(shear) * (1 / self.total_area)
        shear_error = tf.reduce_mean(tf.reduce_sum(shear_error, axis=-1))

        return stretch_loss, stretch_error, shear_loss, shear_error


# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, garment, l, m):
        self.faces = garment.faces
        self.face_areas = garment.face_areas
        self.total_area = garment.surf_area
        self.uv_matrices = garment.uv_matrices
        self.l = l
        self.m = m

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        dX = tf.stack(
            [
                tf.gather(vertices, self.faces[:, 1], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
                tf.gather(vertices, self.faces[:, 2], axis=1)
                - tf.gather(vertices, self.faces[:, 0], axis=1),
            ],
            axis=-1,
        )
        F = dX @ self.uv_matrices
        Ft = tf.linalg.matrix_transpose(F)
        G = 0.5 * (Ft @ F - tf.eye(2))
        S = self.m * G + (0.5 * self.l * tf.einsum("...ii", G))[
            ..., None, None
        ] * tf.eye(2, batch_shape=tf.shape(G)[:2])
        loss = tf.einsum("...ii", tf.linalg.matrix_transpose(S) @ G)
        loss *= self.face_areas
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = loss / (self.total_area)

        return loss, error


class BendingLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_adjacency = garment.face_adjacency
        self.face_adjacency_edges = garment.face_adjacency_edges
        face_areas = garment.face_areas[garment.face_adjacency].sum(-1)
        edge_lengths = garment.face_adjacency_edge_lengths
        self.stiffness_scaling = edge_lengths**2 / (8 * face_areas)
        self.angle_true = garment.face_dir_dihedral

    @tf.function
    def signed_bending_diff(self, vertices, plasticity_target):
        vertices = tf.cast(vertices, tf.float32)
        plasticity_target = tf.cast(plasticity_target, tf.float32)
        mesh_face_normals = face_normals(vertices, self.faces)
        normals0 = tf.gather(mesh_face_normals, self.face_adjacency[:, 0], axis=1)
        normals1 = tf.gather(mesh_face_normals, self.face_adjacency[:, 1], axis=1)

        edge_vertices = tf.gather(vertices, self.face_adjacency_edges, axis=1)
        edge_dir = edge_vertices[:, :, 1] - edge_vertices[:, :, 0]
        edge_dir = tf.math.l2_normalize(edge_dir, axis=-1)

        cos = tf.einsum("abc,abc->ab", normals0, normals1)
        sin = tf.einsum("abc,abc->ab", edge_dir, tf.linalg.cross(normals0, normals1))
        angle = tf.math.atan2(sin, cos) - self.angle_true
        angle = tf.math.atan2(tf.sin(angle), tf.cos(angle))

        return angle - tf.squeeze(plasticity_target)

    @tf.function
    def __call__(self, vertices, plasticity_target):
        diff = self.signed_bending_diff(vertices, plasticity_target)
        loss = diff ** 2
        error = tf.abs(diff)

        loss *= self.stiffness_scaling
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)

        return loss, error


# Fast estimation of SDF
class CollisionLoss:
    def __init__(self, body, collision_threshold=0.004, vertex_weights=None):
        self.body_faces = body.faces
        self.collision_vertices = tf.constant(body.collision_vertices)
        self.collision_threshold = collision_threshold
        self.vertex_weights = (
            None
            if vertex_weights is None
            else tf.constant(vertex_weights[None, :], dtype=tf.float32)
        )

    @tf.function
    def __call__(self, vertices, body_vertices, indices):
        vertices = tf.cast(vertices, tf.float32)
        body_vertices = tf.cast(body_vertices, tf.float32)
        # Compute body vertex normals
        body_vertex_normals = vertex_normals(body_vertices, self.body_faces)
        # Gather collision vertices
        body_vertices = tf.gather(body_vertices, self.collision_vertices, axis=1)
        body_vertex_normals = tf.gather(
            body_vertex_normals, self.collision_vertices, axis=1
        )
        # Compute loss
        cloth_to_body = vertices - tf.gather_nd(body_vertices, indices)
        body_vertex_normals = tf.gather_nd(body_vertex_normals, indices)
        normal_dist = tf.einsum("abc,abc->ab", cloth_to_body, body_vertex_normals)
        loss = tf.minimum(normal_dist - self.collision_threshold, 0.0) ** 2
        if self.vertex_weights is not None:
            loss *= self.vertex_weights
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.math.less(normal_dist, 0.0)
        error = tf.cast(error, tf.float32)
        error = tf.reduce_mean(error)
        return loss, error


class GravityLoss:
    def __init__(self, vertex_area, density=0.15, gravity=[0, 0, -9.81]):
        self.vertex_mass = density * vertex_area[:, None]
        self.gravity = tf.constant(gravity, tf.float32)

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        loss = -self.vertex_mass * vertices * self.gravity
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        return loss


class InertiaLoss:
    def __init__(self, dt, vertex_area, density=0.15):
        self.dt = dt
        self.vertex_mass = density * vertex_area
        self.total_mass = tf.reduce_sum(self.vertex_mass)

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        x0, x1, x2 = tf.unstack(vertices, axis=1)
        x_proj = 2 * x1 - x0
        x_proj = tf.stop_gradient(x_proj)
        dx = x2 - x_proj
        loss = (0.5 / self.dt**2) * self.vertex_mass[:, None] * dx**2
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        error = self.vertex_mass * tf.norm(dx, axis=-1)
        error = tf.reduce_sum(error, axis=-1) / self.total_mass
        error = tf.reduce_mean(error)

        return loss, error


class PinningLoss:
    def __init__(self, garment, pin_blend_weights=False):
        self.indices = garment.pinning_vertices
        self.vertices = garment.vertices[self.indices]
        self.pin_blend_weights = pin_blend_weights
        if pin_blend_weights:
            self.blend_weights = garment.blend_weights[self.indices]

    @tf.function
    def __call__(self, unskinned):
        unskinned = tf.cast(unskinned, tf.float32)
        loss = tf.gather(unskinned, self.indices, axis=-2) - self.vertices
        loss = loss**2
        loss = tf.reduce_sum(loss, axis=[1, 2])
        loss = tf.reduce_mean(loss)
        return loss


class DyBendingLoss:
    def __init__(self, garment):
        self.faces = garment.faces
        self.face_adjacency = garment.face_adjacency
        face_areas = garment.face_areas[garment.face_adjacency].sum(-1)
        edge_lengths = garment.face_adjacency_edge_lengths
        self.stiffness_scaling = edge_lengths**2 / (8 * face_areas)
        self.angle_true = garment.face_dihedral
        self.face_adjacency_edges = garment.face_adjacency_edges

    @tf.function
    def __call__(self, vertices):
        vertices = tf.cast(vertices, tf.float32)
        mesh_face_normals = face_normals(vertices, self.faces)
        normals0 = tf.gather(mesh_face_normals, self.face_adjacency[:, 0], axis=1)
        normals1 = tf.gather(mesh_face_normals, self.face_adjacency[:, 1], axis=1)

        cos = tf.einsum("abc,abc->ab", normals0, normals1)
        sin = tf.norm(tf.linalg.cross(normals0, normals1), axis=-1)
        angle = tf.math.atan2(sin, cos) - self.angle_true

        loss = angle ** 2

        error = tf.abs(angle)
        loss *= self.stiffness_scaling
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)
        return loss, error


class ThresholdLoss:
    def __init__(self, config):
        self.config = config
        # self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def __call__(self, angles, threshold):
        angles_f32 = tf.cast(angles, tf.float32)
        thr_f32 = tf.cast(threshold, tf.float32)
        loss = tf.reduce_mean(tf.square(angles_f32 - thr_f32))
        error = tf.reduce_mean(tf.abs(angles_f32 - thr_f32))

        return loss, error
