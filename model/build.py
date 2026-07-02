from model.ncs import NCS
import tensorflow as tf


def build_ncs_model(config, edge_count):
    model = NCS(config)
    print("Building NCS model...")
    # model.build(input_shape=[*config.input_shape, (None, config.num_time_steps, edge_count, 1)])
    pose_shape, trans_shape = config.input_shape
    dummy_pose = tf.zeros(
        [1, pose_shape[1], pose_shape[2], pose_shape[3]],
        dtype=tf.float32
    )
    dummy_trans = tf.zeros(
        [1, trans_shape[1], trans_shape[2]],
        dtype=tf.float32
    )
    dummy_dec = tf.zeros(
        [1, config.num_time_steps, edge_count, 1],
        dtype=tf.float32
    )
    dummy_phase = tf.zeros(
        [1, config.num_time_steps, 2],
        dtype=tf.float32
    )

    _ = model([dummy_pose, dummy_trans, dummy_dec, dummy_phase], training=False)
    model.summary()
    print("Compiling NCS model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(optimizer=optimizer)
    return model
