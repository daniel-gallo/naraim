from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        normalized_x = nn.LayerNorm()(x)
        return x + self.fn(normalized_x, *args, **kwargs)


class TransformerLayer(nn.Module):
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        attention_residual_block = ResidualBlock(
            nn.Sequential(
                [
                    nn.MultiHeadDotProductAttention(self.num_heads),
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                ]
            )
        )
        mlp_residual_block = ResidualBlock(
            nn.Sequential(
                [
                    nn.Dense(self.hidden_dimension),
                    nn.gelu,
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                    nn.Dense(self.embedding_dimension),
                    nn.Dropout(self.dropout_probability, deterministic=not training),
                ]
            )
        )

        if mask is not None:
            bs, sequence_length, _ = x.shape
            # The mask should be batch_size x num_heads x sequence_length x sequence_length
            # We can use broadcasting for the first two dimensions
            mask = jnp.expand_dims(mask, 0)
            mask = jnp.expand_dims(mask, 0)

        x = attention_residual_block(x, mask=mask)
        x = mlp_residual_block(x)
        return x


class Transformer(nn.Module):
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool, mask=None):
        for _ in range(self.num_layers):
            layer = TransformerLayer(
                num_heads=self.num_heads,
                embedding_dimension=self.embedding_dimension,
                hidden_dimension=self.hidden_dimension,
                dropout_probability=self.dropout_probability,
            )

            x = layer(x, training, mask)

        return x
