import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import load_dataset


def _get_files(split: str):
    snellius_path = Path(
        f"/scratch-shared/fomo_imagenet/tfrecords_imagenet_shuffled_{split}"
    )
    local_path = Path(
        f"/home/robert/Projects/data/imagenet/tfrecords_imagenet_shuffled_{split}"
    )

    for path in (snellius_path, local_path):
        if path.exists():
            return list(path.glob("*.tfrec"))

    raise Exception("No train TFRecords found")


def get_train_files():
    return _get_files("train")


def get_val_files():
    return _get_files("val")


def main():
    train_ds = (
        load_dataset(
            get_train_files(),
            14,
            True,
            False,
        )
        .batch(64)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    wasted = {"percentage": 0, "count": 0}
    for i in train_ds:
        patch_indices = i[1]  # (64, 256, 2)
        wasted_patches = np.where(
            patch_indices == np.array([0, 0]), 1, 0
        )  # (64, 256, 2), entries with [0 0] will become [1 1]
        wasted_patches = np.where(
            np.mean(wasted_patches, axis=-1) > 0.5, 1, 0
        )  # (64, 256), entries with [1 1] will become 1, rest 0
        wasted["percentage"] = (
            (wasted["percentage"] * wasted["count"])
            + (np.sum(wasted_patches, axis=(0, 1)) - i[1].shape[0])
            / (i[1].shape[0] * 256)
            * i[1].shape[0]
        ) / (wasted["count"] + i[1].shape[0])
        wasted["count"] += i[1].shape[0]  # sum all 1s in the matrix
        print(wasted["percentage"])
    print(wasted["percentage"])
    batch = next(train_ds)[1]
    arr1 = np.mean(np.where(next(train_ds)[1] == np.array([0, 0]), 1, 0), axis=-1)

    print(np.sum(np.where(arr1 > 0.5, 1, 0), axis=(0, 1)) - 64)
    print(batch.shape[0] * batch.shape[1])


if __name__ == "__main__":
    main()
