import flax.linen as nn
import jax
from einops import rearrange
from jax import numpy as jnp


def get_1d_positional_embedding(embedding_dimension: int, max_length: int) -> jax.Array:
    """
    The output is an array of shape (max_length, embedding_dimension)
    """
    positions = jnp.arange(max_length)

    dimensions = jnp.arange(embedding_dimension // 2)
    normalized_dimensions = 2 * dimensions / embedding_dimension
    denominator = 10_000**normalized_dimensions

    matrix = positions[:, None] / denominator[None, :]
    sin_embedding = jnp.sin(matrix)
    cos_embedding = jnp.cos(matrix)

    # With concatenation AIM: https://github.com/apple/ml-aim/blob/0b1dea9128f4734ae89252078e65aa102999407a/aim/jax/layers.py#L92
    # embedding = jnp.concatenate((sin_embedding, cos_embedding), axis=1)
    # Interleaving (like in Attention is all you need)
    embedding = rearrange([sin_embedding, cos_embedding], "t l d -> l (d t)")

    return embedding


def get_2d_positional_embedding(
    embedding_dimension: int, height: int, width: int
) -> jax.Array:
    """
    The output is an array of shape (height, width, embedding_dimension)
    """
    assert embedding_dimension % 2 == 0

    height_embedding = get_1d_positional_embedding(embedding_dimension // 2, height)
    width_embedding = get_1d_positional_embedding(embedding_dimension // 2, width)

    return jnp.concatenate(
        [
            height_embedding[:, None, :].repeat(repeats=width, axis=1),
            width_embedding[None, :, :].repeat(repeats=height, axis=0),
        ],
        axis=2,
    )


class PositionalEncoding(nn.Module):
    embedding_dimension: int
    max_num_patches: int

    def setup(self):
        """
        self.positional_encoding: Array of shape (num_patches, num_patches, 2)
        """

        self.positional_encoding = get_2d_positional_embedding(
            embedding_dimension=self.embedding_dimension,
            height=self.max_num_patches,
            width=self.max_num_patches,
        )

    def __call__(self, x, patch_indices):
        """
        Input:
            x: Array of shape (batch_size, num_patches, embedding_dimension)
            patch_indices: Array of shape (batch_size, num_patches, 2)
        Output:
            x: Array of shape (batch_size, num_patches, embedding_dimension)
        """
        positions = self.positional_encoding[
            patch_indices[:, :, 0], patch_indices[:, :, 1]
        ]
        x = x + positions

        return x


class FractionalPositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, patch_indices: jax.Array):
        """
        Input:
            x: Array of shape (batch_size, num_patches, embedding_dimension)
            patch_indices: Array of shape (batch_size, num_patches, 2)
        Output:
            x: Array of shape (batch_size, num_patches, embedding_dimension)
        """
        embedding_dimension = x.shape[-1]

        heights = patch_indices[:, :, 0].max(axis=1)
        widths = patch_indices[:, :, 1].max(axis=1)

        fractional_heights = patch_indices[:, :, 0:1] / heights
        fractional_widths = patch_indices[:, :, 1:2] / widths

        height_embeddings = nn.Dense(embedding_dimension)(fractional_heights)
        widths_embeddings = nn.Dense(embedding_dimension)(fractional_widths)

        return x + height_embeddings + widths_embeddings
