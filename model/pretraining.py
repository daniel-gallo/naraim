import flax.linen as nn
import jax.numpy as jnp

from model.common import InitialProjection
from model.positional import FractionalPositionalEncoding, PositionalEncoding
from model.transformer import Transformer


class PretrainingHead(nn.Module):
    dtype: jnp.dtype
    patch_size: int
    num_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size**2 * self.num_channels, dtype=self.dtype)(x)
        return x


class PretrainingModel(nn.Module):
    dtype: jnp.dtype
    patch_size: int
    max_num_patches: int
    num_channels: int
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float
    use_fractional_positional_encoding: bool = False

    @nn.compact
    def __call__(self, x, patch_indices, is_training: bool, attention_mask=None):
        x = InitialProjection(dtype=self.dtype, embedding_dimension=self.embedding_dimension)(x)
        if self.use_fractional_positional_encoding:
            x = FractionalPositionalEncoding()(x, patch_indices)
        else:
            x = PositionalEncoding(
                embedding_dimension=self.embedding_dimension,
                max_num_patches=self.max_num_patches,
            )(x, patch_indices)
        x = Transformer(
            dtype=self.dtype,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            embedding_dimension=self.embedding_dimension,
            hidden_dimension=self.hidden_dimension,
            dropout_probability=self.dropout_probability,
        )(x, is_training, mask=attention_mask)
        x = PretrainingHead(dtype=self.dtype, patch_size=self.patch_size, num_channels=self.num_channels)(x)

        return x
