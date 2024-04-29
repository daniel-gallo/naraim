from pathlib import Path
from typing import Any, List

import numpy as np
from einops import rearrange
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

DATA_ROOT = "data"


def image_to_numpy(image: Image) -> np.array:
    image = np.array(image, dtype=np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return image


def reshape_image(image: np.array, patch_size: int, max_num_patches: int) -> np.array:
    height, width = image.shape[:2]

    # We want to reshape the image to lambda x height, lambda x width so that
    # the total number of (spacial) pixels is smaller than max_patches x patch_size
    lmbd = np.sqrt(patch_size**2 * max_num_patches / (height * width))
    resize_height = int(lmbd * height)
    resize_width = int(lmbd * width)

    # After resizing the image, the dimensions might not be divisible by patch_size
    # so we have to take a random crop
    crop_height = (resize_height // patch_size) * patch_size
    crop_width = (resize_width // patch_size) * patch_size

    output_image = Compose(
        [
            ToTensor(),
            Resize(size=(resize_height, resize_width)),
            CenterCrop(size=(crop_height, crop_width)),
        ]
    )(image)

    return rearrange(output_image.numpy(), "c h w -> h w c")


def patchify(image: np.array, patch_size: int):
    assert image.shape[0] % patch_size == 0
    assert image.shape[1] % patch_size == 0

    for x_idx in range(image.shape[0] // patch_size):
        for y_idx in range(image.shape[1] // patch_size):
            patch = image[
                x_idx * patch_size : (x_idx + 1) * patch_size,
                y_idx * patch_size : (y_idx + 1) * patch_size,
            ]
            patch_idx = x_idx * (image.shape[1] // patch_size) + y_idx

            yield patch_idx, patch.flatten()


def collate(batch: List, patch_size: int, max_num_patches: int):
    """
    batch: list of (image: np.array<h, w, c>, label: int)
    """
    batch_size = len(batch)
    first_image = batch[0][0]
    if len(first_image.shape) == 2:
        num_channels = 1
    else:
        _, _, num_channels = first_image.shape

    patches = np.zeros((batch_size, max_num_patches, patch_size**2 * num_channels))
    patch_indices = np.zeros((batch_size, max_num_patches, 2), dtype=int)
    labels = np.zeros(batch_size)

    for i, (image, label) in enumerate(batch):
        reshaped_image = reshape_image(image, patch_size, max_num_patches)
        new_height, new_width, _ = reshaped_image.shape

        for patch_idx, patch in patchify(reshaped_image, patch_size):
            patches[i, patch_idx, :] = patch

        sample_indices = rearrange(
            np.meshgrid(
                np.arange(new_height // patch_size),
                np.arange(new_width // patch_size),
                indexing="ij",
            ),
            "t h w -> (h w) t",
        )
        patch_indices[i, : len(sample_indices)] = sample_indices

        labels[i] = label

    return patches, patch_indices, labels


def collate_classification(batch: List, patch_size: int, max_num_patches: int):
    patches, patch_indices, labels = collate(batch, patch_size, max_num_patches)
    return patches, patch_indices, labels


def collate_pretraining(batch: List, patch_size: int, max_num_patches: int):
    patches, patch_indices, _ = collate(batch, patch_size, max_num_patches)
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


def get_fashion_mnist_dataloader(
    pretraining: bool,
    train: bool,
    batch_size: int,
    patch_size: int,
    max_num_patches: int,
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
        collate_fn=lambda batch: collate_fn(batch, patch_size, max_num_patches),
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
    patch_size: int,
    max_num_patches: int,
):
    dataset = get_imagenet_dataset(split)

    collate_fn = collate_pretraining if pretraining else collate_classification
    shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=lambda batch: collate_fn(batch, patch_size, max_num_patches),
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
