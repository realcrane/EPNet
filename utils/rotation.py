from __future__ import annotations

import numpy as np
import tensorflow as tf


@tf.function
def from_axis_angle(axis_angle):
    input_shape = tf.shape(axis_angle)
    tf.debugging.assert_equal(
        input_shape[-1],
        3,
        message="Rotation from axis angle error. Expected last dim = 3.",
    )

    axis_angle_flat = tf.reshape(axis_angle, [-1, 3])
    angle = tf.norm(axis_angle_flat, axis=-1, keepdims=True)
    axis = axis_angle_flat / (angle + tf.keras.backend.epsilon())

    x, y, z = tf.unstack(axis, axis=-1)
    zeros = tf.zeros_like(x)

    skew = tf.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=1,
    )
    skew = tf.reshape(skew, [-1, 3, 3])

    identity = tf.eye(3, batch_shape=[tf.shape(axis_angle_flat)[0]])
    sin_term = tf.sin(angle)[:, None] * skew
    cos_term = (1 - tf.cos(angle))[:, None] * tf.matmul(skew, skew)
    rotations = identity + sin_term + cos_term

    batch_shape = tf.shape(axis_angle)[:-1]
    return tf.reshape(rotations, tf.concat([batch_shape, [3, 3]], axis=0))


@tf.function
def from_quaternion(quaternions):
    input_shape = tf.shape(quaternions)
    tf.debugging.assert_equal(
        input_shape[-1],
        4,
        message="Rotation from quaternion error. Expected last dimension = 4.",
    )

    quaternions = tf.reshape(quaternions, (-1, 4))
    w, x, y, z = tf.unstack(quaternions, axis=-1)

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    rotations = tf.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        axis=-1,
    )

    return tf.reshape(rotations, tf.concat([input_shape[:-1], [3, 3]], axis=0))


def slerp(q0, q1, r):
    r = np.expand_dims(r, axis=[-2, -1])
    dot = np.clip((q0 * q1).sum(-1, keepdims=True), -1, 1)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    w0 = np.empty_like(omega)
    w1 = np.empty_like(omega)
    np.divide(
        np.sin((1 - r) * omega),
        sin_omega,
        out=w0,
        where=np.abs(sin_omega) > np.finfo(np.float32).eps,
    )
    np.divide(
        np.sin(r * omega),
        sin_omega,
        out=w1,
        where=np.abs(sin_omega) > np.finfo(np.float32).eps,
    )
    near_linear = np.abs(sin_omega) <= np.finfo(np.float32).eps
    w0 = np.where(near_linear, 1 - r, w0)
    w1 = np.where(near_linear, r, w1)
    return w0 * q0 + w1 * q1


def axis_angle_to_quat(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1)[..., None] + np.finfo(float).eps
    axis = rotvec / angle
    sin = np.sin(angle / 2)
    w = np.cos(angle / 2)
    return np.concatenate((w, sin * axis), axis=-1)


def quat_to_axis_angle(quat):
    angle = 2 * np.arccos(quat[..., 0:1])
    axis = quat[..., 1:] / (np.sin(angle / 2) + np.finfo(float).eps)
    return angle * axis
