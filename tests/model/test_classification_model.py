import jax.numpy as jnp
from jax import random

from model import ClassificationModel


def test_classification_model():
    dtype = jnp.float32
    bs = 32
    num_patches = 4
    patch_size = 14
    max_num_patches = 256
    num_categories = 10

    num_layers = 8
    num_heads = 4
    embedding_dimension = 768
    hidden_dimension = 128
    dropout_probability = 0.1
    num_channels = 1

    x = jnp.zeros((bs, num_patches, patch_size))
    patch_indices = jnp.zeros((bs, num_patches, 2), dtype=int)

    classification_model = ClassificationModel(
        dtype=dtype,
        max_num_patches=max_num_patches,
        num_categories=num_categories,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_probability,
        patch_size=patch_size,
        num_channels=num_channels,
    )

    rng = random.key(seed=0)
    params = classification_model.init(rng, x, patch_indices, is_training=True)

    # Check if shape is correct
    output_shape = classification_model.apply(params, x, patch_indices, is_training=True, rngs={"dropout": rng}).shape
    assert output_shape == (bs, num_categories)

    # Check inference does not need rng
    classification_model.apply(params, x, patch_indices, is_training=False)
