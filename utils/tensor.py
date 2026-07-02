from __future__ import annotations

import tensorflow as tf


@tf.function
def compute_nth_derivative(x, n, dt):
    for _ in range(n):
        x = (x[:, 1:] - x[:, :-1]) / dt
    return x
