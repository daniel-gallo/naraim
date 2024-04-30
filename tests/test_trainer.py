import jax.numpy as jnp
from jaxlib import xla_extension

from dataset import get_fashion_mnist_dataloader
from trainer import Trainer


def test_autoregressor_training():
    patch_size = 14
    max_num_patches = 256
    num_channels = 1
    lr = 1e-3
    seed = 42
    log_every_n_steps = 10

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
    trainer = Trainer(
        model_type="autoregressor",
        dummy_batch=next(iter(train_dataloader)),
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
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
    assert len(output) == 2

    state, aux_output = output
    loss, rng = aux_output
    assert isinstance(loss, xla_extension.ArrayImpl)
    assert loss.size == 1
    assert isinstance(loss.item(), float)

    # Test eval_step
    mse = trainer.eval_step(trainer.state, next(iter(val_dataloader)))
    assert isinstance(mse, xla_extension.ArrayImpl)
    assert mse.size == 1
    assert isinstance(mse.item(), float)


# @pytest.mark.skip(reason="TrainerClassifier will be deleted")
def test_classifier_training():
    max_num_patches = 256
    num_categories = 10
    lr = 1e-3
    seed = 42
    log_every_n_steps = 10

    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=True, batch_size=4, patch_size=14, max_num_patches=256
    )

    val_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=False, batch_size=4, patch_size=14, max_num_patches=256
    )

    # Test init_model and optimizer
    trainer = Trainer(
        model_type="classifier",
        dummy_batch=next(iter(train_dataloader)),
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        norm_pix_loss=True,
        max_num_patches=max_num_patches,
        dtype=jnp.float32,
        num_categories=num_categories,
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
    assert len(output) == 2
    state, aux_output = output  # state, loss, rng, acc
    loss, rng, acc = aux_output
    assert isinstance(loss, xla_extension.ArrayImpl)
    assert loss.size == 1
    assert isinstance(loss.item(), float)
    assert acc >= 0 and acc <= 1

    # Test eval_step
    acc = trainer.eval_step(trainer.state, next(iter(val_dataloader)))
    assert isinstance(acc, xla_extension.ArrayImpl)
    assert acc.size == 1
    assert isinstance(acc.item(), float)
    assert acc >= 0 and acc <= 1
