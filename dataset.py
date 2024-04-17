import jax.numpy as jnp
import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

DATA_ROOT = "data"


def image_to_numpy(image: Image) -> np.array:
    image = np.array(image, dtype=np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return image


def collate_classification(batch):
    # TODO: this assumes that all images are 28x28 and have 1 channel
    batch_size = len(batch)
    patch_size = 14
    num_patches_per_image = 4

    X = np.zeros((batch_size, num_patches_per_image, patch_size**2), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.int16)

    for i, (image, label) in enumerate(batch):
        assert image.shape == (28, 28)
        X[i, 0, :] = image[:14, :14].flatten()
        X[i, 1, :] = image[:14, 14:].flatten()
        X[i, 2, :] = image[14:, :14].flatten()
        X[i, 3, :] = image[14:, 14:].flatten()

        y[i] = label

    return X, y


def collate_pretraining(batch):
    # TODO: this assumes that all images are 28x28 and have 1 channel
    batch_size = len(batch)
    patch_size = 14
    num_patches_per_image = 3

    X = np.zeros((batch_size, num_patches_per_image, patch_size**2), dtype=np.float32)
    Y = np.zeros((batch_size, num_patches_per_image, patch_size**2), dtype=np.float32)

    for i, (image, _) in enumerate(batch):
        assert image.shape == (28, 28)
        X[i, 0, :] = image[:14, :14].flatten()
        X[i, 1, :] = image[:14, 14:].flatten()
        X[i, 2, :] = image[14:, :14].flatten()

        Y[i, 0, :] = image[:14, 14:].flatten()
        Y[i, 1, :] = image[14:, :14].flatten()
        Y[i, 2, :] = image[14:, 14:].flatten()

    # TODO: is it tril for sure? Or is it triu?
    # TODO: implement uniform sampling
    mask = jnp.tril(jnp.ones((num_patches_per_image, num_patches_per_image)))
    return X, Y, mask


def get_fashion_mnist_dataloader(
    pretraining: bool, train: bool, batch_size: int = 1024
) -> DataLoader:
    dataset = FashionMNIST(
        DATA_ROOT, train=train, transform=image_to_numpy, download=True
    )

    collate_fn = collate_pretraining if pretraining else collate_classification
    shuffle = True if train else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader
