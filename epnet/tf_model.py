from __future__ import annotations

from dataclasses import asdict, dataclass

import tensorflow as tf


@dataclass
class PNetConfig:
    latent_size: int = 128
    output_size: int = 1
    num_layers: int = 2
    n_nodefeatures: int = 10
    n_edgefeatures_mesh: int = 12
    message_passing_steps: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


def make_mlp(widths: list[int], activate_final: bool = False, name: str | None = None):
    layers = []
    for i, width in enumerate(widths[1:]):
        is_final = i == len(widths) - 2
        activation = None if is_final and not activate_final else "relu"
        layers.append(tf.keras.layers.Dense(width, activation=activation))
    return tf.keras.Sequential(layers, name=name)


class MeshMessageBlock(tf.keras.layers.Layer):
    def __init__(self, latent_size: int, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        hidden = [latent_size] * num_layers
        self.edge_mlp = make_mlp([latent_size * 3, *hidden, latent_size], name="edge_mlp")
        self.node_mlp = make_mlp([latent_size * 2, *hidden, latent_size], name="node_mlp")

    def call(self, node_latent, edge_index, edge_latent, training=False):
        senders = edge_index[0]
        receivers = edge_index[1]
        edge_input = tf.concat(
            [
                tf.gather(node_latent, receivers),
                tf.gather(node_latent, senders),
                edge_latent,
            ],
            axis=-1,
        )
        edge_update = self.edge_mlp(edge_input, training=training)
        edge_latent = edge_latent + edge_update

        agg = tf.math.unsorted_segment_sum(
            edge_update,
            receivers,
            num_segments=tf.shape(node_latent)[0],
        )
        node_update = self.node_mlp(tf.concat([node_latent, agg], axis=-1), training=training)
        node_latent = node_latent + node_update
        return node_latent, edge_latent


class PNetMeshOnly(tf.keras.Model):
    def __init__(self, cfg: PNetConfig | dict | None = None, **kwargs):
        super().__init__(**kwargs)
        if cfg is None:
            cfg = PNetConfig()
        elif isinstance(cfg, dict):
            cfg = PNetConfig(**{k: v for k, v in cfg.items() if k in PNetConfig.__dataclass_fields__})
        self.cfg = cfg

        hidden = [cfg.latent_size] * cfg.num_layers
        self.node_encoder = make_mlp(
            [cfg.n_nodefeatures, *hidden, cfg.latent_size],
            name="node_encoder",
        )
        self.edge_encoder = make_mlp(
            [cfg.n_edgefeatures_mesh, *hidden, cfg.latent_size],
            name="edge_encoder",
        )
        self.blocks = [
            MeshMessageBlock(cfg.latent_size, cfg.num_layers, name=f"block_{i}")
            for i in range(cfg.message_passing_steps)
        ]
        self.decoder = make_mlp([cfg.latent_size, *hidden, cfg.output_size], name="decoder")

    def call(self, inputs, training=False):
        node_features, edge_index, edge_features = inputs
        node_latent = self.node_encoder(node_features, training=training)
        edge_latent = self.edge_encoder(edge_features, training=training)
        for block in self.blocks:
            node_latent, edge_latent = block(
                node_latent,
                edge_index,
                edge_latent,
                training=training,
            )
        return self.decoder(node_latent, training=training)


def make_pnet_model(cfg_dict: dict | None = None) -> tuple[PNetMeshOnly, dict]:
    cfg = PNetConfig(**{k: v for k, v in (cfg_dict or {}).items() if k in PNetConfig.__dataclass_fields__})
    model = PNetMeshOnly(cfg)
    dummy_node = tf.zeros((1, cfg.n_nodefeatures), tf.float32)
    dummy_edge_index = tf.zeros((2, 1), tf.int32)
    dummy_edge = tf.zeros((1, cfg.n_edgefeatures_mesh), tf.float32)
    model((dummy_node, dummy_edge_index, dummy_edge), training=False)
    final_layer = model.decoder.layers[-1]
    final_layer.kernel.assign(tf.zeros_like(final_layer.kernel))
    final_layer.bias.assign(tf.zeros_like(final_layer.bias))
    return model, cfg.to_dict()
