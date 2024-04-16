import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import block_diag

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


def test_mask():
    """
    Test the following mask:
    [1 1 0 0]
    [1 1 0 0]
    [0 0 1 1]
    [0 0 1 1]

    Given an input [x1, x2, x1, x2], the output should be [y1, y2, y1, y2] (if dropout is disabled)
    """
    bs = 1
    n = 4
    d = 768

    model = Transformer(
        num_layers=8,
        num_heads=6,
        embedding_dimension=768,
        hidden_dimension=128,
        dropout_probability=0.1,
    )
    rng = random.key(seed=0)

    half_x = random.normal(rng, (bs, n // 2, d))
    x = jnp.concat([half_x, half_x], axis=1)
    assert x.shape == (bs, n, d)

    mask = block_diag(*[jnp.ones((n // 2, n // 2))] * 2)
    assert mask.shape == (n, n)

    params = model.init(rng, x, training=False)
    output = model.apply(params, x, training=False, mask=mask, rngs={"dropout": rng})
    assert jnp.all(jnp.isclose(output[:, : n // 2, :], output[:, n // 2 :, :]))


# TODO: create a test for the autoregressive framework. Do we need jnp.tril, right? or jnp.triu? #dyslexia
