import jax.numpy as jnp
from jax import random

from model import ClassificationModel


def test_classification_model():
    dtype = jnp.float32
    bs = 32
    num_patches = 4
    patch_size = 196
    num_categories = 10

    num_layers = 8
    num_heads = 4
    embedding_dimension = 768
    hidden_dimension = 128
    dropout_probability = 0.1

    x = jnp.zeros((bs, num_patches, patch_size))

    classification_model = ClassificationModel(
        dtype=dtype,
        num_categories=num_categories,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_probability,
    )

    rng = random.key(seed=0)
    params = classification_model.init(rng, x, training=True)
    output_shape = classification_model.apply(
        params, x, training=True, rngs={"dropout": rng}
    ).shape

    assert output_shape == (bs, num_categories)
