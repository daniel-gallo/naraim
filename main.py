import argparse

import jax.numpy as jnp

from dataset import get_dataloader
from trainer import TrainerAutoregressor, TrainerClassifier


def train_autoregressor(
    batch_size,
    num_epochs,
    embedding_dimension,
    hidden_dimension,
    num_layers,
    num_heads,
    patch_size,
    dataset_name="fashion_mnist",
):
    train_dataloader = get_dataloader(
        dataset_name, pretraining=True, train=True, batch_size=batch_size
    )

    val_dataloader = get_dataloader(
        dataset_name, pretraining=True, train=False, batch_size=batch_size
    )

    trainer = TrainerAutoregressor(
        dummy_imgs=next(iter(train_dataloader))[0],
        norm_pix_loss=True,
        patch_size=patch_size,
        dtype=jnp.bfloat16,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=0.1,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=num_epochs)
    val_mse = trainer.eval_model(val_dataloader)
    print(f"Final MSE: {val_mse}")


def train_classifier(
    batch_size,
    num_epochs,
    embedding_dimension,
    hidden_dimension,
    num_layers,
    num_heads,
    num_categories,
    dataset_name="fashion_mnist",
):
    train_dataloader = get_dataloader(
        dataset_name, pretraining=False, train=True, batch_size=batch_size
    )

    val_dataloader = get_dataloader(
        dataset_name, pretraining=False, train=False, batch_size=batch_size
    )

    trainer = TrainerClassifier(
        dummy_imgs=next(iter(train_dataloader))[0],
        dtype=jnp.bfloat16,
        num_categories=num_categories,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=0.1,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=num_epochs)
    val_acc = trainer.eval_model(val_dataloader)
    print(f"Final accuracy: {val_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument(
        "--train_classifier", action="store_true", help="Train classifier"
    )
    parser.add_argument(
        "--train_autoregressor", action="store_true", help="Train autoregressor"
    )

    parser.add_argument(
        "--num_categories",
        type=int,
        default=10,
        help="Number of classification categories",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=768,
        help="Embedding dimension of the tokens",
    )
    parser.add_argument(
        "--hidden_dimension",
        type=int,
        default=128,
        help="Hidden dimension of the model",
    )
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument("--patch_size", type=int, default=196, help="Patch size")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["fashion_mnist", "imagenet"],
        help="Dataset name",
    )

    args = parser.parse_args()

    train_classifier(
        args.batch_size,
        args.epochs,
        args.embedding_dimension,
        args.hidden_dimension,
        args.num_layers,
        args.num_heads,
        args.num_categories,
        dataset_name=args.dataset,
    )
    train_autoregressor(
        args.batch_size,
        args.epochs,
        args.embedding_dimension,
        args.hidden_dimension,
        args.num_layers,
        args.num_heads,
        args.patch_size,
        dataset_name=args.dataset,
    )
