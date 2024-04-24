from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.datasets.folder import ImageFolder

DATA_ROOT = "data"


def image_to_numpy(image: Image) -> np.array:
    image = np.array(image, dtype=np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return image


def collate_classification(
    batch, n_channels=3, patch_size=14, padding=0.0, num_patches_per_image=0
):
    batch_size = len(batch)

    if not num_patches_per_image:
        num_patches = [
            (image.shape[0] // patch_size) * (image.shape[1] // patch_size)
            for image, _ in batch
        ]
        num_patches_per_image = max(num_patches)

    patched_images = np.full(
        shape=(batch_size, num_patches_per_image, patch_size**2 * n_channels),
        fill_value=padding,
        dtype=np.float32,
    )

    labels = np.zeros(batch_size, dtype=np.int16)
    resolutions = np.zeros((batch_size, 2), dtype=np.int16)

    for i, (image, label) in enumerate(batch):
        for x_idx in range(image.shape[0] // patch_size):
            for y_idx in range(image.shape[1] // patch_size):
                patch = image[
                    x_idx * patch_size : (x_idx + 1) * patch_size,
                    y_idx * patch_size : (y_idx + 1) * patch_size,
                ]
                patch_idx = x_idx * (image.shape[1] // patch_size) + y_idx

                patched_images[i, patch_idx, :] = patch.flatten()

        labels[i] = label
        resolutions[i] = image.shape[:2]

    return patched_images, labels, resolutions


def collate_pretraining(batch, n_channels=3, patch_size=14, padding=0.0):
    patched_images, labels, resolutions = collate_classification(
        batch, n_channels, patch_size, padding
    )
    patched_images = patched_images[:, :-1, ...]
    num_patches_per_image = patched_images.shape[1]

    # TODO: is it tril for sure? Or is it triu?
    # TODO: implement uniform sampling
    mask = jnp.tril(jnp.ones((num_patches_per_image, num_patches_per_image)))
    return patched_images, labels, mask, resolutions


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
        collate_fn=lambda batch: collate_fn(batch, n_channels=1),
    )

    return dataloader


class ImageNet(ImageFolder):
    IMAGENET_ROOT = "/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/"
    WNID_TO_CLASSES_FILE = "/scratch-nvme/ml-datasets/imagenet/LOC_synset_mapping.txt"

    def __init__(
        self,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        self.root = Path(self.IMAGENET_ROOT)
        assert split in ["train", "val", "test"]

        self.split_folder = self.root / split
        wnid_to_classes = self.load_wnid_to_classes_file(
            Path(self.WNID_TO_CLASSES_FILE)
        )

        super().__init__(self.split_folder, **kwargs)

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }

    def load_wnid_to_classes_file(self, path):
        lines = path.read_text().split("\n")
        wnid_to_classes = dict()
        for line in lines:
            sep = line.find(" ")
            wnid = line[:sep]
            classes = line[sep + 1 :]
            wnid_to_classes[wnid] = classes

        return wnid_to_classes


def get_imagenet_dataset(
    split: str,
):
    dataset = ImageNet(split, transform=image_to_numpy)

    return dataset


def get_imagenet_dataloader(
    pretraining: bool,
    split: str,
    batch_size: int,
):
    dataset = get_imagenet_dataset(split)

    collate_fn = collate_pretraining if pretraining else collate_classification
    shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader


def get_dataloader(
    dataset: str,
    *args,
    **kwargs,
):
    if dataset == "fashion_mnist":
        return get_fashion_mnist_dataloader(*args, **kwargs)
    elif dataset == "imagenet":
        return get_imagenet_dataloader(*args, **kwargs)
    else:
        raise ValueError(f"Dataset {dataset} does not exist")
