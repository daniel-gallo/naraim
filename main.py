from dataset import get_fashion_mnist_dataloader
from trainer import TrainerAutoregressor, TrainerClassifier
import jax.numpy as jnp


def train_autoregressor():
    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=True, train=True, batch_size=1024
    )

    val_dataloader = get_fashion_mnist_dataloader(
        pretraining=True, train=False, batch_size=1024
    )

    trainer = TrainerAutoregressor(
        dummy_imgs=next(iter(train_dataloader))[0],
        norm_pix_loss=True,
        patch_size=196,
        dtype=jnp.bfloat16,
        num_layers=8,
        num_heads=6,
        embedding_dimension=768,
        hidden_dimension=128,
        dropout_probability=0.1,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=2)
    val_mse = trainer.eval_model(val_dataloader)
    print(f"Final MSE: {val_mse}")


def train_classifier():
    train_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=True, batch_size=1024
    )

    val_dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=False, batch_size=1024
    )

    trainer = TrainerClassifier(
        dummy_imgs=next(iter(train_dataloader))[0],
        dtype=jnp.bfloat16,
        num_categories=10,
        num_layers=8,
        num_heads=4,
        embedding_dimension=768,
        hidden_dimension=128,
        dropout_probability=0.1,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=2)
    val_acc = trainer.eval_model(val_dataloader)
    print(f"Final accuracy: {val_acc}")


# TODO: Argparser
# TODO: Making the methods more generic (adding some arguments for the two functions lol)
if __name__ == "__main__":
    train_classifier()
    train_autoregressor()
