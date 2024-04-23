import flax.linen as nn
import jax.numpy as jnp

from transformer import Transformer


class InitialProjection(nn.Module):
    dtype: jnp.dtype
    embedding_dimension: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.embedding_dimension, dtype=self.dtype)(x)


class PositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x):
        # TODO: implement
        return x


class PretrainingHead(nn.Module):
    dtype: jnp.dtype
    patch_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size, dtype=self.dtype)(x)
        return x


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
            (batch_size, 1, embedding_dimension),
        )
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype)(
            inputs_q=q, inputs_k=x, inputs_v=x
        )
        x = x.squeeze()
        x = nn.Dense(self.num_categories, dtype=self.dtype)(x)

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


class ClassificationModel(nn.Module):
    dtype: jnp.dtype
    num_categories: int
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool):
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
        )(x, training)
        x = ClassificationHead(
            dtype=self.dtype,
            num_heads=self.num_heads,
            num_categories=self.num_categories,
        )(x)

        return x
