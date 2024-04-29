from functools import wraps
from pathlib import Path

import numpy as np
import pytest

from dataset import (
    collate_classification,
    collate_pretraining,
    get_fashion_mnist_dataloader,
    get_imagenet_dataloader,
    reshape_image,
)


@pytest.mark.parametrize("height", [100, 224, 500])
@pytest.mark.parametrize("width", [100, 224, 500])
@pytest.mark.parametrize("patch_size", [14, 16])
@pytest.mark.parametrize("max_num_patches", [256, 512])
def test_reshape_img(height: int, width: int, patch_size: int, max_num_patches: int):
    image = np.zeros((height, width, 3))
    reshaped_image = reshape_image(
        image, patch_size=patch_size, max_num_patches=max_num_patches
    )

    assert isinstance(reshaped_image, np.ndarray)
    assert reshaped_image.dtype == np.float64

    new_height, new_width, num_channels = reshaped_image.shape
    assert new_height % patch_size == 0
    assert new_width % patch_size == 0
    assert num_channels == 3


def test_collate_classification():
    batch = [
        (np.zeros((14, 14, 3)), 0),
        (np.zeros((28, 28, 3)), 0),
        (np.zeros((42, 42, 3)), 0),
        (np.zeros((100, 200, 3)), 0),
    ]
    patch_size = 14
    max_num_patches = 256

    patches, patch_indices, labels = collate_classification(
        batch, patch_size, max_num_patches
    )

    assert patches.shape == (len(batch), max_num_patches, patch_size**2 * 3)
    assert patch_indices.shape == (len(batch), max_num_patches, 2)
    assert labels.shape == (len(batch),)

    # The first image is square, so it should have 16 x 16 patches
    assert np.all(patch_indices[0][0] == [0, 0])
    assert np.all(patch_indices[0][1] == [0, 1])
    assert np.all(patch_indices[0][-1] == [15, 15])

    # The last image does not fit perfectly, so the last patches should be padding
    assert np.all(patch_indices[-1][0] == [0, 0])
    assert np.all(patch_indices[-1][1] == [0, 1])
    assert np.all(patch_indices[-1][-1] == [0, 0])


def test_collate_pretraining():
    batch = [
        (np.zeros((14, 14, 3)), 0),
        (np.zeros((28, 28, 3)), 0),
        (np.zeros((42, 42, 3)), 0),
        (np.zeros((100, 200, 3)), 0),
    ]
    batch_size = len(batch)
    num_channels = 3
    patch_size = 14
    max_num_patches = 256

    input_patches, attention_mask, loss_mask, patch_indices, output_patches = (
        collate_pretraining(batch, patch_size, max_num_patches)
    )

    assert input_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert output_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert attention_mask.shape == (max_num_patches - 1, max_num_patches - 1)
    assert loss_mask.shape == (batch_size, max_num_patches - 1)
    assert patch_indices.shape == (batch_size, max_num_patches - 1, 2)

    # The attention mask should be tril + square on the lop left corner
    prefix_length = np.argmin(attention_mask[0])
    expected_attention_mask = np.tril(
        np.ones((max_num_patches - 1, max_num_patches - 1))
    )
    expected_attention_mask[:prefix_length, :prefix_length] = 1
    assert np.allclose(expected_attention_mask, attention_mask)

    # The loss mask should hide the prefix...
    assert np.allclose(loss_mask[:, :prefix_length], 0)
    # ... and the padding (the last image does not fit perfectly)
    assert np.allclose(loss_mask[-1, -13:], 0)
    # (the rest should be ones)
    assert np.allclose(loss_mask[:, prefix_length:-13], 1)
    assert np.allclose(loss_mask[:-1, prefix_length:], 1)


# TODO: FIXME
def ignore_if_not_on_snellius(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not Path("/scratch-nvme/ml-datasets/imagenet").exists():
            return

        return f(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_pretraining(train: bool):
    batch_size = 1024
    patch_size = 14
    max_num_patches = 256
    num_channels = 1

    dataloader = get_fashion_mnist_dataloader(
        pretraining=True,
        train=train,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )
    input_patches, attention_mask, loss_mask, patch_indices, output_patches = next(
        iter(dataloader)
    )

    assert input_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert output_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert attention_mask.shape == (max_num_patches - 1, max_num_patches - 1)
    assert loss_mask.shape == (batch_size, max_num_patches - 1)
    assert patch_indices.shape == (batch_size, max_num_patches - 1, 2)


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_classification(train: bool):
    batch_size = 1024
    patch_size = 14
    max_num_patches = 256
    num_channels = 1

    dataloader = get_fashion_mnist_dataloader(
        pretraining=False,
        train=train,
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )
    patches, patch_indices, labels = next(iter(dataloader))
    assert patches.shape == (batch_size, max_num_patches, patch_size**2 * num_channels)
    assert labels.shape == (batch_size,)
    assert patch_indices.shape == (batch_size, max_num_patches, 2)


@ignore_if_not_on_snellius
def test_imagenet_dataloader_pretraining():
    batch_size = 4
    patch_size = 14
    max_num_patches = 256
    num_channels = 3

    dataloader = get_imagenet_dataloader(
        pretraining=True,
        split="train",
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    input_patches, attention_mask, loss_mask, patch_indices, output_patches = next(
        iter(dataloader)
    )

    assert input_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert output_patches.shape == (
        batch_size,
        max_num_patches - 1,
        patch_size**2 * num_channels,
    )
    assert attention_mask.shape == (max_num_patches - 1, max_num_patches - 1)
    assert loss_mask.shape == (batch_size, max_num_patches - 1)
    assert patch_indices.shape == (batch_size, max_num_patches - 1, 2)


@ignore_if_not_on_snellius
def test_imagenet_dataloader_classification():
    batch_size = 4
    patch_size = 14
    max_num_patches = 256
    num_channels = 3

    dataloader = get_imagenet_dataloader(
        pretraining=False,
        split="train",
        batch_size=batch_size,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )

    patches, patch_indices, labels = next(iter(dataloader))
    assert patches.shape == (batch_size, max_num_patches, patch_size**2 * num_channels)
    assert labels.shape == (batch_size,)
    assert patch_indices.shape == (batch_size, max_num_patches, 2)
