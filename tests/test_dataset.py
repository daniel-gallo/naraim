import pytest

from dataset import get_fashion_mnist_dataloader


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_pretraining(train: bool):
    batch_size = 1024
    num_patches = 3
    patch_size = 14 * 14

    dataloader = get_fashion_mnist_dataloader(
        pretraining=True, train=train, batch_size=batch_size
    )
    X, Y, mask = next(iter(dataloader))

    assert X.shape == (batch_size, num_patches, patch_size)
    assert Y.shape == (batch_size, num_patches, patch_size)
    assert mask.shape == (num_patches, num_patches)
    print(mask)


@pytest.mark.parametrize("train", [True, False])
def test_fashion_mnist_classification(train: bool):
    batch_size = 1024
    num_patches = 4
    patch_size = 14 * 14

    dataloader = get_fashion_mnist_dataloader(
        pretraining=False, train=train, batch_size=batch_size
    )
    X, y = next(iter(dataloader))
    assert X.shape == (batch_size, num_patches, patch_size)
    assert y.shape == (batch_size,)
