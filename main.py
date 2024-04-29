import argparse

import jax.numpy as jnp

from dataset import get_dataloader
from trainer import TrainerAutoregressor, TrainerClassifier


def train_autoregressor(
    batch_size: int,
    num_epochs: int,
    lr: float,
    seed: int,
    log_every_n_steps: int,
    norm_pix_loss: bool,
    embedding_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    patch_size: int,
    max_num_patches: int,
    num_channels: int,
    dataset_name: str,
):
    train_dataloader = get_dataloader(
        dataset_name,
        pretraining=True,
        train=True,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    val_dataloader = get_dataloader(
        dataset_name,
        pretraining=True,
        train=False,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    trainer = TrainerAutoregressor(
        dummy_batch=next(iter(train_dataloader)),
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        norm_pix_loss=norm_pix_loss,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        num_channels=num_channels,
        dtype=jnp.float32,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        dropout_probability=dropout_rate,
    )

    trainer.train_model(train_dataloader, val_dataloader, num_epochs=num_epochs)
    val_mse = trainer.eval_model(val_dataloader)
    print(f"Final MSE: {val_mse}")


def train_classifier(
    batch_size: int,
    num_epochs: int,
    lr: float,
    seed: int,
    log_every_n_steps: int,
    embedding_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    num_heads: int,
    num_categories: int,
    patch_size: int,
    max_num_patches: int,
    dataset_name: str,
):
    train_dataloader = get_dataloader(
        dataset_name,
        pretraining=False,
        train=True,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    val_dataloader = get_dataloader(
        dataset_name,
        pretraining=False,
        train=False,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    trainer = TrainerClassifier(
        dummy_batch=next(iter(train_dataloader)),
        lr=lr,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        dtype=jnp.float32,
        max_num_patches=max_num_patches,
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
        "--training",
        type=str,
        default="autoregressor",
        choices=["classifier", "autoregressor"],
        help="What to train",
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
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument(
        "--max_num_patches", type=int, default=256, help="Max number of patches"
    )

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--norm_pix_loss",
        type=bool,
        default=True,
        help="Whether to use a normalized-pixel loss",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps until next logging",
    )

    parser.add_argument("--num_channels", type=int, default=1, help="Num channels")

    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion_mnist",
        choices=["fashion_mnist", "imagenet"],
        help="Dataset name",
    )

    args = parser.parse_args()

    if args.training == "autoregressor":
        train_autoregressor(
            args.batch_size,
            args.epochs,
            args.lr,
            args.seed,
            args.log_every_n_steps,
            args.norm_pix_loss,
            args.embedding_dimension,
            args.hidden_dimension,
            args.num_layers,
            args.num_heads,
            args.dropout_rate,
            args.patch_size,
            args.max_num_patches,
            args.num_channels,
            args.dataset,
        )

    elif args.training == "classifier":
        train_classifier(
            args.batch_size,
            args.epochs,
            args.lr,
            args.seed,
            args.log_every_n_steps,
            args.embedding_dimension,
            args.hidden_dimension,
            args.num_layers,
            args.num_heads,
            args.num_categories,
            args.patch_size,
            args.max_num_patches,
            args.dataset,
        )
