import jax.numpy as jnp
from jax import random
import jax

from transformer import ResidualBlock, TransformerLayer, Transformer


def test_residual_block():
    bs = 3
    n = 18
    d = 768
    x = jnp.zeros((bs, n, d))

    model = ResidualBlock(fn=lambda x: x)
    rng = random.key(seed=0)
    params = model.init(rng, x)
    output_shape = model.apply(params, x).shape

    assert output_shape == (bs, n, d)


def test_transformer_layer():
    bs = 3
    n = 18
    d = 768
    x = jnp.zeros((bs, n, d))

    model = TransformerLayer(
        num_heads=6,
        embedding_dimension=768,
        hidden_dimension=128,
        dropout_probability=0.1,
    )
    rng = random.key(seed=0)
    params = model.init(rng, x, training=True)
    output_shape = model.apply(params, x, training=True, rngs={"dropout": rng}).shape

    assert output_shape == (bs, n, d)


def test_transformer():
    bs = 3
    n = 18
    d = 768
    x = jnp.zeros((bs, n, d))

    model = Transformer(
        num_layers=8,
        num_heads=6,
        embedding_dimension=768,
        hidden_dimension=128,
        dropout_probability=0.1,
    )
    rng = random.key(seed=0)
    params = model.init(rng, x, training=True)
    output_shape = model.apply(params, x, training=True, rngs={"dropout": rng}).shape

    assert output_shape == (bs, n, d)
    print(jax.tree_map(lambda x: x.shape, params))
