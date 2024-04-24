import flax.linen as nn
import jax.numpy as jnp


class InitialProjection(nn.Module):
    dtype: jnp.dtype
    embedding_dimension: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.embedding_dimension, dtype=self.dtype)(x)
