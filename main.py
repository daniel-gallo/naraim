import argparse
import os
import subprocess
import sys
from pathlib import Path

import tensorflow as tf

from dataset import load_dataset, prefetch
from trainer_DP import Trainer


def simulate_CPU_devices(device_count: int = 4):
    # Set XLA flags to simulate a CPU with a given number of devices
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    # Disable CUDA to force XLA to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Check for packages to be installed if needed. On Colab, the following packages are not installed by default:
    # - ml_collections


#
#
# def install_package(package: str) -> None:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


def _get_files(split: str):
    snellius_path = Path(
        f"/scratch-shared/fomo_imagenet/tfrecords_imagenet_shuffled_{split}"
    )
    local_path = Path(f"./tfrecords_imagenet_shuffled_{split}")

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

    model_args.add_argument(
        "--num_categories", type=int, help="Number of classes for classifier"
    )
    model_args.add_argument("--use_fractional_positional_encoding", action="store_true")


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
        "--lr_end_value",
        type=float,
        default=0.0,
        help="Learning rate at the end of cosine decay",
    )
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
        "--load_only_params",
        action="store_true",
        default=False,
        help="Load only model params from the checkpoint",
    )

    trainer_args.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Freeze the backbone (InitialProjection and Transformer)",
    )

    trainer_args.add_argument(
        "--norm_pix_loss",
        type=str,
        default="True",
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

    trainer_args.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    trainer_args.add_argument(
        "--n_images_to_visualize", type=int, default=5, help="How many images to plot"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")

    model_args = parser.add_argument_group("model")
    add_model_args(model_args)

    trainer_args = parser.add_argument_group("trainer")
    add_trainer_args(trainer_args)

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--native_resolutions",
        action="store_true",
        help="Whether we use the native resolutions of the images",
    )
    parser.add_argument(
        "--should_apply_auto_augment",
        action="store_true",
        help="Whether we should apply AutoAugment",
    )

    args = parser.parse_args()

    model_hparams = {
        a.dest: getattr(args, a.dest, None) for a in model_args._group_actions
    }
    trainer_kwargs = {
        a.dest: getattr(args, a.dest, None) for a in trainer_args._group_actions
    }

    if trainer_kwargs["model_type"] == "autoregressor":
        model_hparams.pop("num_categories")

    return args, model_hparams, trainer_kwargs


if __name__ == "__main__":
    simulate_CPU_devices()

    args, model_hparams, trainer_kwargs = parse_args()

    print(f"Training with native resolutions: {args.native_resolutions}")

    if args.norm_pix_loss == "False":
        args.norm_pix_loss = False
    else:
        args.norm_pix_loss = True

    print(f"Using normalized pixel-loss: {args.norm_pix_loss}")

    train_ds = prefetch(
        load_dataset(
            get_train_files(),
            args.patch_size,
            args.native_resolutions,
            args.should_apply_auto_augment,
        )
        .shuffle(4 * args.batch_size)
        .repeat()
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    validation_ds = prefetch(
        load_dataset(
            get_val_files(),
            args.patch_size,
            args.native_resolutions,
            args.should_apply_auto_augment,
        )
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    args.dummy_batch = next(iter(train_ds))
    trainer_kwargs["dummy_batch"] = next(iter(train_ds))
    trainer = Trainer(model_hparams=model_hparams, **trainer_kwargs)
    trainer.train_model(train_ds, validation_ds, args.max_num_iterations)
