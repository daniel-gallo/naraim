import jax.numpy as jnp
from jax import random

from model import PretrainingModel


def test_pretraining_model():
    dtype = jnp.float32
    bs = 32
    num_patches = 4
    patch_size = 14
    max_num_patches = 256

    num_layers = 8
    num_heads = 4
    embedding_dimension = 768
    hidden_dimension = 128
    dropout_probability = 0.1
    num_channels = 1

    x = jnp.zeros((bs, num_patches, patch_size))
    patch_indices = jnp.zeros((bs, num_patches, 2), dtype=int)

    pretraining_model = PretrainingModel(
        dtype=dtype,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        num_channels=num_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_probability,
    )

    rng = random.key(seed=0)
    params = pretraining_model.init(rng, x, patch_indices, is_training=True)

    # Check if shape is correct
    output_shape = pretraining_model.apply(params, x, patch_indices, is_training=True, rngs={"dropout": rng}).shape
    assert output_shape == (bs, num_patches, patch_size**2 * num_channels)

    # Check inference does not need rng
    pretraining_model.apply(params, x, patch_indices, is_training=False)
