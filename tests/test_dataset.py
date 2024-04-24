import numpy as np
import pytest

from dataset import (
    collate_classification,
    collate_pretraining,
    get_fashion_mnist_dataloader,
    get_imagenet_dataloader,
)


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_pretraining(train: bool):
    batch_size = 1024
    num_patches = 3
    patch_size = 14 * 14

    dataloader = get_fashion_mnist_dataloader(
        pretraining=True, train=train, batch_size=batch_size
    )
    X, Y, mask, resolutions = next(iter(dataloader))

    assert X.shape == (batch_size, num_patches, patch_size)
    assert Y.shape == (batch_size,)
    assert mask.shape == (num_patches, num_patches)
    assert resolutions.shape == (batch_size, 2)

    print(mask)


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_classification(train: bool):
    batch_size = 1024
    num_patches = 4
    patch_size = 14 * 14

    dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=train, batch_size=batch_size
    )
    X, y, resolutions = next(iter(dataloader))
    assert X.shape == (batch_size, num_patches, patch_size)
    assert y.shape == (batch_size,)
    assert resolutions.shape == (batch_size, 2)


def test_collate_classification():
    batch = [
        (np.zeros((14, 14, 3)), 0),
        (np.zeros((28, 28, 3)), 0),
        (np.zeros((42, 42, 3)), 0),
        (np.zeros((250, 250, 3)), 0),
    ]

    patched_images, labels, resolutions = collate_classification(batch)

    assert patched_images.shape == (4, 289, 588)
    assert labels.shape == (4,)
    assert resolutions.shape == (4, 2)
    assert np.all(resolutions == [[14, 14], [28, 28], [42, 42], [250, 250]])


def test_collate_pretraining():
    batch = [
        (np.zeros((14, 14, 3)), 0),
        (np.zeros((28, 28, 3)), 0),
        (np.zeros((42, 42, 3)), 0),
        (np.zeros((250, 250, 3)), 0),
    ]

    patched_images, labels, mask, resolutions = collate_pretraining(batch)

    assert patched_images.shape == (4, 288, 588)
    assert labels.shape == (4,)
    assert resolutions.shape == (4, 2)
    assert mask.shape == (288, 288)
    assert np.all(resolutions == [[14, 14], [28, 28], [42, 42], [250, 250]])


@pytest.mark.parametrize("train", [True, False])
def test_imagenet_dataloader_pretraining(train: bool):
    batch_size = 64
    patch_size = 14 * 14 * 3

    dataloader = get_imagenet_dataloader(
        pretraining=True, split="train", batch_size=batch_size
    )

    X, Y, mask = next(iter(dataloader))

    assert X.shape[0] == batch_size
    assert X.shape[2] == patch_size
    assert Y.shape == (batch_size,)


@pytest.mark.parametrize("train", [True, False])
def test_imagenet_dataloader_classification(train: bool):
    batch_size = 64
    patch_size = 14 * 14 * 3

    dataloader = get_imagenet_dataloader(
        pretraining=False, split="train", batch_size=batch_size
    )

    X, Y, _ = next(iter(dataloader))

    assert X.shape[0] == batch_size
    assert X.shape[2] == patch_size
    assert Y.shape == (batch_size,)
