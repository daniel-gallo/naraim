import argparse
from pathlib import Path

import tensorflow as tf

from dataset import load_dataset, prefetch
from trainer import Trainer


def _get_files(split: str):
    snellius_path = Path(f"/scratch-shared/fomo_imagenet/tfrecords_imagenet_{split}")
    local_path = Path("./tfrecords")

    for path in (snellius_path, local_path):
        if path.exists():
            return list(path.glob("*.tfrec"))

    raise Exception("No train TFRecords found")


def get_train_files():
    return _get_files("train")


def get_val_files():
    return _get_files("val")


def add_model_args(model_args: argparse._ArgumentGroup):
    """
    Arguments that will be passed to initialize the model
    """
    model_args.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"]
    )
    model_args.add_argument("--patch_size", type=int, default=14, help="Patch size")
    model_args.add_argument(
        "--max_num_patches", type=int, default=256, help="Max number of patches"
    )
    model_args.add_argument("--num_channels", type=int, default=3, help="Num channels")
    model_args.add_argument(
        "--num_layers", type=int, default=12, help="Number of layers"
    )
    model_args.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    model_args.add_argument(
        "--embedding_dimension",
        type=int,
        default=768,
        help="Embedding dimension of the tokens",
    )
    model_args.add_argument(
        "--hidden_dimension",
        type=int,
        default=3072,
        help="Hidden dimension of the MLP of the ViT",
    )
    model_args.add_argument(
        "--dropout_probability", type=float, default=0.0, help="Dropout rate"
    )


def add_trainer_args(trainer_args: argparse._ArgumentGroup):
    """
    Arguments that will be used to initialize the Trainer
    """
    trainer_args.add_argument(
        "--model_type",
        type=str,
        default="autoregressor",
        choices=["classifier", "autoregressor"],
        help="What to train",
    )
    trainer_args.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    trainer_args.add_argument(
        "--beta2", type=float, default=0.98, help="Beta2 parameter for AdamW"
    )
    trainer_args.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer",
    )
    trainer_args.add_argument("--seed", type=int, default=42, help="Seed")
    trainer_args.add_argument(
        "--log_every_n_steps",
        type=int,
        default=200,
        help="Number of steps in between logging",
    )
    trainer_args.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=10_000,
        help="Number of steps in between evals",
    )

    trainer_args.add_argument(
        "--checkpoints_path",
        type=str,
        required=True,
        help="Path in which the checkpoints will be saved",
    )

    trainer_args.add_argument(
        "--tensorboard_path",
        type=str,
        required=True,
        help="Path in which tensorboard files will be saved",
    )

    trainer_args.add_argument(
        "--checkpoint_path_to_load",
        type=str,
        help="Path of the checkpoint to be loaded",
    )

    trainer_args.add_argument(
        "--norm_pix_loss",
        type=bool,
        default=True,
        help="Whether to use a normalized-pixel loss",
    )

    trainer_args.add_argument(
        "--max_num_iterations",
        type=int,
        required=True,
        help="Maximum number of iterations",
    )
    trainer_args.add_argument("--profile", action="store_true")

    trainer_args.add_argument("--warmup_steps", type=int, default=5_000)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")

    model_args = parser.add_argument_group("model")
    add_model_args(model_args)

    trainer_args = parser.add_argument_group("trainer")
    add_trainer_args(trainer_args)

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    args = parser.parse_args()

    model_hparams = {
        a.dest: getattr(args, a.dest, None) for a in model_args._group_actions
    }
    trainer_kwargs = {
        a.dest: getattr(args, a.dest, None) for a in trainer_args._group_actions
    }

    return args, model_hparams, trainer_kwargs


if __name__ == "__main__":
    args, model_hparams, trainer_kwargs = parse_args()

    train_ds = prefetch(
        load_dataset(get_train_files(), args.patch_size)
        .shuffle(4 * args.batch_size)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )

    validation_ds = prefetch(
        load_dataset(get_val_files(), args.patch_size)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    args.dummy_batch = next(iter(train_ds))

    trainer_kwargs["dummy_batch"] = next(iter(train_ds))
    trainer = Trainer(model_hparams=model_hparams, **trainer_kwargs)
    trainer.train_model(train_ds, validation_ds, args.max_num_iterations)
