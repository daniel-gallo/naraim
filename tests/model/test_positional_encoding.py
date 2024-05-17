import jax
import jax.random
import pytest
from jax import numpy as jnp

from model.positional import FractionalPositionalEncoding, get_2d_positional_embedding


def test_positional_encoding_with_odd_embedding_dimension_fails():
    embedding_dimension = 127
    height = 12
    width = 8

    with pytest.raises(AssertionError):
        get_2d_positional_embedding(embedding_dimension, height, width)


def test_positional_encoding():
    embedding_dimension = 128
    height = 12
    width = 8

    encoding = get_2d_positional_embedding(embedding_dimension, height, width)
    assert encoding.shape == (height, width, embedding_dimension)


def test_fractional_encoding():
    batch_size = 1
    num_patches = 9
    embedding_dimension = 64

    x = jnp.zeros((batch_size, num_patches, embedding_dimension))
    patch_indices = jnp.array(
        [
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            ]
        ]
    )

    fractional_encoding = FractionalPositionalEncoding()
    params = fractional_encoding.init(jax.random.key(0), x, patch_indices)
    y = fractional_encoding.apply(params, x, patch_indices)
    assert y.shape == x.shape
