from typing import Callable

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
    def __call__(self, x, training: bool):
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

        x = attention_residual_block(x)
        x = mlp_residual_block(x)
        return x


class Transformer(nn.Module):
    num_layers: int
    num_heads: int
    embedding_dimension: int
    hidden_dimension: int
    dropout_probability: float

    @nn.compact
    def __call__(self, x, training: bool):
        for _ in range(self.num_layers):
            layer = TransformerLayer(
                num_heads=self.num_heads,
                embedding_dimension=self.embedding_dimension,
                hidden_dimension=self.hidden_dimension,
                dropout_probability=self.dropout_probability,
            )

            x = layer(x, training)

        return x
