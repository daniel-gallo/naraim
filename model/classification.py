import flax.linen as nn
import jax.numpy as jnp

from model.common import InitialProjection
from model.positional import FractionalPositionalEncoding, PositionalEncoding
from model.transformer import Transformer


class ClassificationHead(nn.Module):
    dtype: jnp.dtype
    num_heads: int
    num_categories: int

    @nn.compact
    def __call__(self, x):
        """This is an attentive probe: see AIM pape (2401.08541)"""
        # TODO: is this really an attentive probe, or should we add some bells and whistles (ResBlock...)
        batch_size, _, embedding_dimension = x.shape
        q = self.param(
            "probe_query",
            nn.initializers.lecun_normal(),
            (1, 1, embedding_dimension),
        )
        q_batched = jnp.tile(q, (batch_size, 1, 1))
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype)(
            inputs_q=q_batched, inputs_k=x, inputs_v=x
        )
        x = x.squeeze()
        x = nn.Dense(self.num_categories, dtype=self.dtype)(x)

        return x


class ClassificationModel(nn.Module):
    dtype: jnp.dtype
    patch_size: int
    max_num_patches: int
    num_channels: int
    num_categories: int
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float
    use_fractional_positional_encoding: bool = False

    @nn.compact
    def __call__(self, x, patch_indices, training: bool):
        x = InitialProjection(
            dtype=self.dtype, embedding_dimension=self.embedding_dimension
        )(x)
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
        )(x, training)
        x = ClassificationHead(
            dtype=self.dtype,
            num_heads=self.num_heads,
            num_categories=self.num_categories,
        )(x)

        return x


class NoTransformerClassificationModel(nn.Module):
    dtype: jnp.dtype
    patch_size: int
    max_num_patches: int
    num_channels: int
    num_categories: int
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float
    use_fractional_positional_encoding: bool = False

    @nn.compact
    def __call__(self, x, patch_indices, training: bool):
        x = InitialProjection(
            dtype=self.dtype, embedding_dimension=self.embedding_dimension
        )(x)
        x = ClassificationHead(
            dtype=self.dtype,
            num_heads=self.num_heads,
            num_categories=self.num_categories,
        )(x)

        return x
