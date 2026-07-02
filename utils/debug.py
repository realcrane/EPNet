from __future__ import annotations

import tensorflow as tf

from utils.mesh import face_normals


def angle_to_threshold(angles, step_smooth_scale=1.0, angle_step=0):
    num_steps = tf.shape(angles)[1]
    threshold_prev = tf.zeros_like(angles[:, 0, :, :])
    thresholds = tf.TensorArray(dtype=tf.float32, size=num_steps + 1)
    thresholds = thresholds.write(0, threshold_prev)

    def cond(t, *_):
        return t <= num_steps

    def body(t, threshold_prev, values):
        diff = angles[:, t - 1, :, :] - threshold_prev
        update = tf.sigmoid(step_smooth_scale * (diff - angle_step))
        threshold = update * (diff - angle_step) + threshold_prev
        values = values.write(t, threshold)
        return t + 1, threshold, values

    _, _, thresholds = tf.while_loop(
        cond,
        body,
        loop_vars=[1, threshold_prev, thresholds],
    )

    thresholds = thresholds.stack()
    thresholds = tf.transpose(thresholds, [1, 0, 2, 3])
    return thresholds[:, :-1, :, :]


def signed_angle_to_threshold(angles, step_smooth_scale=1.0, angle_step=0):
    num_steps = tf.shape(angles)[1]
    threshold_prev = tf.zeros_like(angles[:, 0, :, :])
    thresholds = tf.TensorArray(dtype=tf.float32, size=num_steps + 1)
    thresholds = thresholds.write(0, threshold_prev)

    def cond(t, *_):
        return t <= num_steps

    def body(t, threshold_prev, values):
        diff = angles[:, t - 1, :, :] - threshold_prev
        magnitude = tf.abs(diff)
        update = tf.sigmoid(step_smooth_scale * (magnitude - angle_step))
        threshold = threshold_prev + tf.sign(diff) * update * tf.maximum(magnitude - angle_step, 0.0)
        values = values.write(t, threshold)
        return t + 1, threshold, values

    _, _, thresholds = tf.while_loop(
        cond,
        body,
        loop_vars=[1, threshold_prev, thresholds],
    )

    thresholds = thresholds.stack()
    thresholds = tf.transpose(thresholds, [1, 0, 2, 3])
    return thresholds[:, :-1, :, :]


def deformation_to_angle(deformation, garment):
    mesh_face_normals = face_normals(deformation, garment.faces)
    normals0 = tf.gather(mesh_face_normals, garment.face_adjacency[:, 0], axis=2)
    normals1 = tf.gather(mesh_face_normals, garment.face_adjacency[:, 1], axis=2)

    cos = tf.einsum("abcd,abcd->abc", normals0, normals1)
    sin = tf.norm(tf.linalg.cross(normals0, normals1), axis=-1)
    return tf.math.atan2(sin, cos)


def deformation_to_signed_angle(deformation, garment):
    mesh_face_normals = face_normals(deformation, garment.faces)
    normals0 = tf.gather(mesh_face_normals, garment.face_adjacency[:, 0], axis=2)
    normals1 = tf.gather(mesh_face_normals, garment.face_adjacency[:, 1], axis=2)

    edge_vertices = tf.gather(deformation, garment.face_adjacency_edges, axis=2)
    edge_dir = edge_vertices[:, :, :, 1] - edge_vertices[:, :, :, 0]
    edge_dir = tf.math.l2_normalize(edge_dir, axis=-1)

    cos = tf.einsum("abcd,abcd->abc", normals0, normals1)
    sin = tf.einsum("abcd,abcd->abc", edge_dir, tf.linalg.cross(normals0, normals1))
    return tf.math.atan2(sin, cos)
