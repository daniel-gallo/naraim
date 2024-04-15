import flax.linen as nn
from jax import random

from dataset import get_fashion_mnist_dataloader
from transformer import Transformer


class InitialProjection(nn.Module):
    @nn.compact
    def __call__(self, x):
        # TODO: implement
        return x


class PositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x):
        # TODO: implement
        return x


class PretrainingHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        # TODO: implement
        return x


class ClassificationHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        # TODO: implement
        return x


class PretrainingModel(nn.Module):
    transformer: Transformer

    @nn.compact
    def __call__(self, x, training: bool):
        x = InitialProjection()(x)
        x = PositionalEncoding()(x)
        x = self.transformer(x, training)
        x = PretrainingHead()(x)

        return x


class ClassificationModel(nn.Module):
    transformer: Transformer

    @nn.compact
    def __call__(self, x, training: bool):
        x = InitialProjection()(x)
        x = PositionalEncoding()(x)
        x = self.transformer(x, training)
        x = ClassificationHead()(x)

        return x


if __name__ == "__main__":
    dataloader = get_fashion_mnist_dataloader(pretraining=True, train=True)
    transformer = Transformer(
        num_layers=8,
        num_heads=4,
        embedding_dimension=196,
        hidden_dimension=128,
        dropout_probability=0.1,
    )
    pretraining_model = PretrainingModel(transformer=transformer)

    x, y = next(iter(dataloader))
    rng = random.key(seed=0)
    params = pretraining_model.init(rng, x, training=True)
    output_shape = pretraining_model.apply(
        params, x, training=True, rngs={"dropout": rng}
    ).shape

    print(x.shape)
    print(y.shape)
    print(output_shape)
