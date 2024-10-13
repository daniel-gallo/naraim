import argparse
from pathlib import Path

import tensorflow as tf

from dataset import load_dataset, prefetch
from trainer import Trainer
from transformations import (
    AIMInference,
    AutoAugment,
    NativeAspectRatioResize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    SquareResize,
)
from transformations.transformation import Transformation


def _get_files(split: str, dataset_path: str):
    dataset_path = Path(dataset_path) / split
    if dataset_path.exists():
        return list(dataset_path.glob("*.tfrec"))

    raise Exception("No train TFRecords found")


def get_train_files(dataset_path: str):
    return _get_files("train", dataset_path)


def get_val_files(dataset_path: str):
    return _get_files("val", dataset_path)


def get_transformation(transformation: str) -> Transformation:
    if transformation == "auto_augment":
        return AutoAugment()
    elif transformation == "native_aspect_ratio_resize":
        return NativeAspectRatioResize(square_size=224, patch_size=14)
    elif transformation == "random_horizontal_flip":
        return RandomHorizontalFlip()
    elif transformation == "random_resized_crop":
        return RandomResizedCrop(size=224, scale=(0.4, 1.0), ratio=(0.75, 1.33))
    elif transformation == "random_crop":
        return RandomCrop(scale=(0.4, 1.0), ratio=(0.75, 1.33), min_num_pixels=224 * 224)
    elif transformation == "square_resize":
        return SquareResize(224)
    elif transformation == "aim_inference":
        return AIMInference(resize_size=256, crop_size=224)
    else:
        raise NotImplementedError()


def assert_transformation_list_is_valid(transformation_list: [Transformation]):
    # FIXME #24: AutoAugment only works if it's the last transformation
    num_transformations = len(transformation_list)

    for i, transformation in enumerate(transformation_list):
        if isinstance(transformation, AutoAugment) and i != (num_transformations - 1):
            raise NotImplementedError("AutoAugment has to be the last transformation. ")


def add_model_args(model_args: argparse._ArgumentGroup):
    """
    Arguments that will be passed to initialize the model
    """
    model_args.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    model_args.add_argument("--patch_size", type=int, default=14, help="Patch size")
    model_args.add_argument("--max_num_patches", type=int, default=256, help="Max number of patches")
    model_args.add_argument("--num_channels", type=int, default=3, help="Num channels")
    model_args.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    model_args.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
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
    model_args.add_argument("--dropout_probability", type=float, default=0.0, help="Dropout rate")

    model_args.add_argument("--num_categories", type=int, help="Number of classes for classifier")
    model_args.add_argument("--use_fractional_positional_encoding", action="store_true")


def add_trainer_args(trainer_args: argparse._ArgumentGroup):
    """
    Arguments that will be used to initialize the Trainer
    """
    trainer_args.add_argument(
        "--model_type",
        type=str,
        default="autoregressor",
        choices=["classifier", "autoregressor", "no_transformer_classifier"],
        help="What to train",
    )
    trainer_args.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    trainer_args.add_argument(
        "--lr_end_value",
        type=float,
        default=0.0,
        help="Learning rate at the end of cosine decay",
    )
    trainer_args.add_argument("--beta2", type=float, default=0.98, help="Beta2 parameter for AdamW")
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
        default=500,
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
        "--cooldown_steps",
        type=int,
        default=10_000,
        help="Cooldown number of iterations for the exponential lr schedule",
    )

    trainer_args.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
        help="Decay rate for the exponential lr schedule",
    )

    trainer_args.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm")

    trainer_args.add_argument("--n_images_to_visualize", type=int, default=5, help="How many images to plot")

    trainer_args.add_argument(
        "--num_minibatches",
        type=int,
        default=1,
        help="Number of minibatches to use for gradient accumulation",
    )

    trainer_args.add_argument(
        "--lr_schedule_type",
        type=str,
        choices=["exponential", "cosine"],
        default="exponential",
        help="Type of learning rate schedule",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")

    model_args = parser.add_argument_group("model")
    add_model_args(model_args)

    trainer_args = parser.add_argument_group("trainer")
    add_trainer_args(trainer_args)

    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--train_transformations", nargs="+", required=True)
    parser.add_argument("--validation_transformations", nargs="+", required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()

    model_hparams = {a.dest: getattr(args, a.dest, None) for a in model_args._group_actions}
    trainer_kwargs = {a.dest: getattr(args, a.dest, None) for a in trainer_args._group_actions}

    if trainer_kwargs["model_type"] == "autoregressor":
        model_hparams.pop("num_categories")

    return args, model_hparams, trainer_kwargs


if __name__ == "__main__":
    args, model_hparams, trainer_kwargs = parse_args()

    train_transformations = list(map(get_transformation, args.train_transformations))
    validation_transformations = list(map(get_transformation, args.validation_transformations))
    assert_transformation_list_is_valid(train_transformations)
    assert_transformation_list_is_valid(validation_transformations)

    train_ds = prefetch(
        load_dataset(get_train_files(args.dataset_path), args.patch_size, train_transformations)
        .shuffle(4 * args.batch_size)
        .repeat()
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    validation_ds = (
        load_dataset(
            get_val_files(args.dataset_path),
            args.patch_size,
            validation_transformations,
        )
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    args.dummy_batch = next(iter(train_ds))

    trainer_kwargs["dummy_batch"] = next(iter(train_ds))
    trainer = Trainer(model_hparams=model_hparams, **trainer_kwargs)
    trainer.train_model(train_ds, validation_ds)
