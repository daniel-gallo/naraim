import flax.linen as nn
import jax.numpy as jnp

from model.common import InitialProjection
from model.positional import PositionalEncoding
from model.transformer import Transformer


class PretrainingHead(nn.Module):
    dtype: jnp.dtype
    patch_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size, dtype=self.dtype)(x)
        return x


class PretrainingModel(nn.Module):
    dtype: jnp.dtype
    patch_size: int
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        x = InitialProjection(
            dtype=self.dtype, embedding_dimension=self.embedding_dimension
        )(x)
        x = PositionalEncoding()(x)
        x = Transformer(
            dtype=self.dtype,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=self.hidden_dimension,
            dropout_probability=self.dropout_probability,
        )(x, training, mask=mask)
        x = PretrainingHead(dtype=self.dtype, patch_size=self.patch_size)(x)

        return x