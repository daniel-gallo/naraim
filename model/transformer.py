from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    dtype: jnp.dtype
    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        normalized_x = nn.LayerNorm(dtype=self.dtype)(x)
        return x + self.fn(normalized_x, *args, **kwargs)


def dot_product_attention(query, key, value, mask):
    # Like nn.dot_product_attention but uses float32 before softmax (Karras et al., 2023)
    # Taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/single_gpu_transformer.html
    num_features = query.shape[-1]
    dtype = query.dtype
    scale = num_features**-0.5
    query = query * scale
    # Switch dtype right before the dot-product for numerical stability.
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(jnp.float32).min)
    weights = nn.softmax(weights, axis=-1)
    # After softmax, switch back to the original dtype
    weights = weights.astype(dtype)
    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    new_vals = new_vals.astype(dtype)
    return new_vals


class AttentionBlock(nn.Module):
    dtype: jnp.dtype
    num_heads: int

    @nn.compact
    def __call__(self, x, mask=None):
        input_features = x.shape[-1]

        qkv = nn.DenseGeneral(
            features=(self.num_heads, input_features * 3),
            dtype=self.dtype,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = dot_product_attention(q, k, v, mask)
        x = nn.DenseGeneral(
            input_features, axis=(-2, -1), dtype=self.dtype, name="output_layer"
        )(x)
        return x


class TransformerLayer(nn.Module):
    dtype: jnp.dtype
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        attention_residual_block = ResidualBlock(
            dtype=self.dtype,
            fn=nn.remat(nn.Sequential)(
                [
                    AttentionBlock(dtype=self.dtype, num_heads=self.num_heads),
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                ]
            ),
        )
        mlp_residual_block = ResidualBlock(
            dtype=self.dtype,
            fn=nn.remat(nn.Sequential)(
                [
                    nn.Dense(self.hidden_dimension, dtype=self.dtype),
                    nn.gelu,
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                    nn.Dense(self.embedding_dimension, dtype=self.dtype),
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                ]
            ),
        )

        # TODO: only do this if mask has two dimensions
        if mask is not None:
            bs, sequence_length, _ = x.shape
            # The mask should be batch_size x num_heads x sequence_length x sequence_length
            # We can use broadcasting for the first two dimensions
            # mask = jnp.expand_dims(mask, 0)
            mask = jnp.expand_dims(mask, 1)

        x = attention_residual_block(x, mask=mask)
        x = mlp_residual_block(x)
        return x


class Transformer(nn.Module):
    dtype: jnp.dtype
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        for _ in range(self.num_layers):
            layer = TransformerLayer(
                dtype=self.dtype,
                num_heads=self.num_heads,
                embedding_dimension=self.embedding_dimension,
                hidden_dimension=self.hidden_dimension,
                dropout_probability=self.dropout_probability,
            )

            x = layer(x, training, mask)

        return x
