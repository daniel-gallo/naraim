import jax.numpy as jnp
import pytest

from common import are_nested_dicts_equal
from dataset import get_fashion_mnist_dataloader
from trainer import Trainer


@pytest.mark.parametrize("step", [3, 5])
def test_checkpointing(step):
    patch_size = 14
    max_num_patches = 256
    num_channels = 1
    lr = 1e-3
    seed = 42
    log_every_n_steps = 10
    log_dir = "testing"

    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=True,
        train=True,
        batch_size=4,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    trainer = Trainer(
        model_type="autoregressor",
        dummy_batch=next(iter(train_dataloader)),
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        log_dir=log_dir,
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

    # Test whether the saved and loaded model parameters from the same step are the same
    trainer.init_optimizer()
    trainer.save_model(step=step)
    params_saved_model = trainer.state.params

    trainer.load_model(step=step)
    params_loaded_model = trainer.state.params
    assert are_nested_dicts_equal(params_saved_model, params_loaded_model) is True
