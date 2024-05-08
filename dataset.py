import os
from glob import glob
from pathlib import Path
from typing import Any, List

import numpy as np
from einops import rearrange
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.datasets.folder import ImageFolder

DATA_ROOT = "data"

import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def resize_and_crop_image(image, patch_size=16):
    # extract the true height and width of the image (they are None when implicit)
    H, W, C = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
    # compute the sqrt of the aspect ratio
    sqrt_ratio = tf.cast(tf.sqrt(H / W), tf.float32)
    # compute the new height and width
    H = tf.cast(224 * sqrt_ratio, tf.int32)
    W = tf.cast(224**2 / tf.cast(H, tf.float32), tf.int32)
    # resize the image, now the num pixels is ~= 224^2
    image = tf.image.resize(image, [H, W])

    h, w = tf.shape(image)[0], tf.shape(image)[1]
    target_height = h - (h % patch_size)
    target_width = w - (w % patch_size)
    offset_height = (h - target_height) // 2
    offset_width = (w - target_width) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width
    )
    return image


def patchify(image, patch_size=16):
    H, W, C = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
    # calculate the number of patches for this specific image
    P_h = tf.cast(H / patch_size, tf.int32)
    P_w = tf.cast(W / patch_size, tf.int32)
    # crop away any pixels that don't fit into patches
    image = image[: P_h * patch_size, : P_w * patch_size]
    # [P_h, patch_size, P_w, patch_size, C]
    image = tf.reshape(image, [P_h, patch_size, P_w, patch_size, C])
    # [P_h, P_w, patch_size, patch_size, C]
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    # [P_h*P_w, patch_size, patch_size, C]
    image = tf.reshape(image, [-1, patch_size, patch_size, C])
    # seq_len = P_h*P_w
    seq_len = tf.shape(image)[0]
    # [seq_len, patch_dim] where patch_dim = patch_size*patch_size*C
    image = tf.reshape(image, [seq_len, -1])

    image_coords = tf.stack(
        [
            tf.repeat(tf.range(P_h, dtype=tf.int32), (P_w,)),
            tf.tile(tf.range(P_w, dtype=tf.int32), (P_h,)),
        ],
        axis=-1,
    )
    return image, image_coords


def read_labeled_tfrecord(example, patch_size=16):
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature)
    image = decode_image(example["image"])
    image = resize_and_crop_image(image, patch_size)
    image, image_coords = patchify(image, patch_size)

    seq_len = (224 // patch_size) ** 2
    image, padding_mask = pad_sequence(image, seq_len)
    image_coords, _ = pad_sequence(image_coords, seq_len)
    label = tf.cast(example["class_id"], tf.int32)
    return image, image_coords, label


def pad_sequence(seq, seq_len):
    unpadded_seq_len = tf.shape(seq)[0]
    seq_dim = tf.shape(seq)[1]
    padding_mask_false = tf.zeros(unpadded_seq_len, dtype=tf.bool)
    padding_mask_true = tf.ones(seq_len - unpadded_seq_len, dtype=tf.bool)
    padding_mask = tf.concat([padding_mask_false, padding_mask_true], axis=0)

    padding_length = seq_len - unpadded_seq_len
    padding_seq = tf.zeros([padding_length, seq_dim], dtype=seq.dtype)
    seq = tf.concat([seq, padding_seq], axis=0)

    return seq, padding_mask


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset_len = sum(1 for _ in dataset)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset, dataset_len


def get_training_dataset(filenames, batch_size):
    dataset, dataset_len = load_dataset(filenames)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset, dataset_len // batch_size


def get_val_dataset(filenames, batch_size):
    dataset, dataset_len = load_dataset(filenames)
    dataset = dataset.batch(512)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset, dataset_len // batch_size


def collate_pretraining(patches, patch_indices, max_num_patches: int):
    input_patches = patches[:, :-1, :]
    patch_indices = patch_indices[:, :-1, :]
    output_patches = patches[:, 1:, :]

    # TODO: is it tril for sure? Or is it triu?
    # TODO: should prefix-sampling be per-batch of per-sample?
    # We have to subtract two in total:
    #  - 1 because the last patch is not passed
    #  - 1 because we want to run gradient descent on at least one patch
    #      (we run gradient descent in between the prefix and the padding)
    num_patches = (patch_indices.max(axis=1) + 1).prod(axis=1)
    first_dimension_with_padding = num_patches.min() - 1
    prefix_length = np.random.randint(low=1, high=first_dimension_with_padding - 1)
    attention_mask = np.tril(np.ones((max_num_patches - 1, max_num_patches - 1)))
    attention_mask[:prefix_length, :prefix_length] = 1

    loss_mask = np.ones((len(input_patches), max_num_patches - 1))
    loss_mask[:, :prefix_length] = 0

    for i, idx in enumerate(num_patches):
        loss_mask[i, idx:] = 0

    return input_patches, attention_mask, loss_mask, patch_indices, output_patches


if __name__ == "__main__":
    batch_size = 512
    image_dir = "./tfrecords_imagenet_"
    train_files = glob(os.path.join(image_dir + "train", "*.tfrec"))
    val_files = glob(os.path.join(image_dir + "val", "*.tfrec"))
    train_dataset, num_train_batches = get_training_dataset(train_files, batch_size)
    val_dataset, num_val_batches = get_val_dataset(val_files, batch_size)
    train_ds = (
        train_dataset.shuffle(10 * batch_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )
    val_ds = (
        val_dataset.batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )
