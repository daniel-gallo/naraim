import jax
import jax.numpy as jnp
import pytest
from jaxlib import xla_extension

from dataset import get_fashion_mnist_dataloader
from trainer import TrainerAutoregressor, TrainerClassifier


def test_autoregressor_training():
    patch_size = 14
    max_num_patches = 256
    num_channels = 1

    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=True,
        train=True,
        batch_size=4,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    val_dataloader = get_fashion_mnist_dataloader(
        pretraining=True,
        train=False,
        batch_size=4,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    # Test init_model and optimizer
    trainer = TrainerAutoregressor(
        dummy_batch=next(iter(train_dataloader)),
        norm_pix_loss=True,
        patch_size=14,
        max_num_patches=max_num_patches,
        num_channels=num_channels,
        dtype=jnp.float32,
        num_layers=1,
        num_heads=1,
        embedding_dimension=128,
        hidden_dimension=128,
        dropout_probability=0.1,
    )
    assert trainer.state is None
    trainer.init_optimizer()
    assert trainer.state is not None

    # Test train_step
    output = trainer.train_step(
        trainer.state, trainer.rng, next(iter(train_dataloader))
    )
    assert len(output) == 3
    _, _, loss = output  # state, rng, loss
    assert isinstance(loss, xla_extension.ArrayImpl)
    assert loss.size == 1
    assert isinstance(loss.item(), float)

    # Test eval_step
    rng = jax.random.PRNGKey(42)
    mse = trainer.eval_step(trainer.state, rng, next(iter(val_dataloader)))
    assert isinstance(mse, xla_extension.ArrayImpl)
    assert mse.size == 1
    assert isinstance(mse.item(), float)


# TODO: test the new generic Trainer
@pytest.mark.skip(reason="TrainerClassifier will be deleted")
def test_classifier_training():
    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=True, batch_size=4, patch_size=14, max_num_patches=256
    )

    val_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=False, batch_size=4, patch_size=14, max_num_patches=256
    )

    trainer = TrainerClassifier(
        dummy_imgs=next(iter(train_dataloader))[0],
        dtype=jnp.bfloat16,
        num_categories=10,
        num_layers=1,
        num_heads=1,
        embedding_dimension=128,
        hidden_dimension=128,
        dropout_probability=0.1,
    )

    assert trainer.state is None
    trainer.init_optimizer()
    assert trainer.state is not None

    # Test train_step
    output = trainer.train_step(
        trainer.state, trainer.rng, next(iter(train_dataloader))
    )
    assert len(output) == 4
    _, _, loss, acc = output  # state, rng, loss, acc
    assert isinstance(loss, xla_extension.ArrayImpl)
    assert loss.size == 1
    assert isinstance(loss.item(), float)
    assert acc >= 0 and acc <= 1

    # Test eval_step
    rng = jax.random.PRNGKey(42)
    acc = trainer.eval_step(trainer.state, rng, next(iter(val_dataloader)))
    assert isinstance(acc, xla_extension.ArrayImpl)
    assert acc.size == 1
    assert isinstance(acc.item(), float)
    assert acc >= 0 and acc <= 1
