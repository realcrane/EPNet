from __future__ import annotations

import tensorflow as tf
from keras.layers import Layer
from scipy.spatial import cKDTree

from utils.mesh import lbs
from utils.rotation import from_axis_angle, from_quaternion


class FullyConnected(Layer):
    def __init__(self, units, act=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.act = act or (lambda x: x)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

    def call(self, x):
        x = tf.expand_dims(x, axis=-2)
        x = x @ self.kernel
        x = x[..., 0, :]
        if self.use_bias:
            x += self.bias
        return self.act(x)


class SkelFlatten(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def call(self, inputs):
        static_shape = inputs.shape
        if static_shape[-2] is not None and static_shape[-1] is not None:
            flat_dim = static_shape[-2] * static_shape[-1]
            batch_shape = tf.shape(inputs)[:-2]
            return tf.reshape(inputs, tf.concat([batch_shape, [flat_dim]], axis=0))

        input_shape = tf.shape(inputs)
        batch_shape = input_shape[:-2]
        return tf.reshape(inputs, tf.concat([batch_shape, [-1]], axis=0))


class PSD(Layer):
    def __init__(self, num_verts, num_dims=3, act=None, **kwargs):
        super().__init__(**kwargs)
        self.num_verts = num_verts
        self.num_dims = num_dims
        self.act = act or (lambda x: x)

    def build(self, input_shape):
        shape = input_shape[-1], self.num_verts, self.num_dims
        self.psd = tf.Variable(tf.initializers.glorot_normal()(shape), name="psd")

    def call(self, x):
        psd = tf.cast(self.psd, x.dtype)
        x = tf.tensordot(x, psd, axes=[[-1], [0]])
        return self.act(x)


class Skeleton(Layer):
    def __init__(self, rest_joints, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.rest_joints = rest_joints

    def call(self, matrices):
        return lbs(self.rest_joints, matrices)


class LBS(Layer):
    def __init__(self, blend_weights, trainable, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        if trainable:
            blend_weights = tf.math.log(blend_weights + 0.001)
        self.blend_weights = self.add_weight(
            shape=blend_weights.shape,
            initializer=tf.keras.initializers.Constant(blend_weights),
            trainable=trainable,
            name="blend_weights",
        )

    def call(self, vertices, matrices):
        if self.trainable:
            blend_weights = tf.nn.softmax(tf.convert_to_tensor(self.blend_weights))
            return lbs(vertices, matrices, blend_weights)
        blend_weights = tf.convert_to_tensor(self.blend_weights)
        return lbs(vertices, matrices, blend_weights)


class Rotation(Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def build(self, input_shape):
        if input_shape[-1] == 3:
            self.mode = "axis_angle"
        elif input_shape[-1] == 4:
            self.mode = "quaternion"
        else:
            raise ValueError(f"Unsupported rotation input shape: {input_shape}")

    def call(self, orientations):
        if self.mode == "axis_angle":
            return from_axis_angle(orientations)
        return from_quaternion(orientations)


class Collision(Layer):
    def __init__(self, body, use_ray=False, **kwargs):
        super().__init__(trainable=False, dynamic=True, **kwargs)
        self.collision_vertices = tf.constant(body.collision_vertices)
        self.use_ray = use_ray
        self.run_sample = lambda elem: cKDTree(elem[1]).query(elem[0], workers=-1)[1]
        if use_ray:
            import ray

            ray.init()
            self.ray = ray
            self.run_sample = ray.remote(self.run_sample)

    def run(self, vertices, collider):
        if self.use_ray:
            return self.ray.get(
                [self.run_sample.remote(elem) for elem in zip(vertices, collider)]
            )
        return tf.stack([self.run_sample(elem) for elem in zip(vertices, collider)])

    def build(self, input_shape):
        batch_size, num_verts = input_shape[:2]
        self.idx = tf.tile(tf.range(batch_size)[:, None], [1, num_verts])

    def call(self, vertices, collider):
        batch_size = tf.shape(vertices)[0]
        nearest = self.run(vertices, tf.gather(collider, self.collision_vertices, axis=-2))
        return tf.stack(
            [tf.gather(self.idx, tf.range(batch_size), axis=0), nearest],
            axis=-1,
        )
