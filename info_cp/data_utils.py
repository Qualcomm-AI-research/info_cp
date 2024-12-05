# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


"""Provides functions for data preprocessing."""


from os import PathLike
from typing import Tuple, Union

from PIL import Image
from torch import LongTensor, Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

cifar10_transform_train = Compose(
    [
        RandomHorizontalFlip(),
        RandomCrop(32, padding=4, pad_if_needed=True, fill=121, padding_mode="constant"),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
    ]
)


cifar10_transform_test = Compose(
    [
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
    ]
)


cifar100_transform_train = Compose(
    [
        RandomHorizontalFlip(),
        RandomCrop(32, padding=4, pad_if_needed=True, fill=121, padding_mode="constant"),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
    ]
)


cifar100_transform_test = Compose(
    [
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)),
    ]
)


class EMNIST(datasets.EMNIST):
    """
    We override torchvision's EMNIST class to avoid casting the target to int in the __getitem__
    function. This is to facilitate side information experiments (see scripts/side_information.py).
    """

    def __getitem__(self, index: int):
        """
        Fetch the image and labels (including the coarse label) for the data point indexed by index.

        Parameters
        ----------
        index int
            Index of the data point to be fetched.

        Returns
        -------
        tuple: (PIL Image, NDArray)
            Where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_mnist_train_test_datasets(
    dataset_root: Union[str, PathLike],
) -> Tuple[Dataset[tuple[Tensor, LongTensor]], Dataset[tuple[Tensor, LongTensor]]]:
    """
    Fetch MNIST train and test dataset.

    Parameters
    ----------
    dataset_root: str
        The path where the data is to be downloaded.

    Returns
    -------
    dataset: Dataset
        The training dataset.
    test_dataset: Dataset
        The test dataset.
    """
    transform = Compose([ToTensor(), Lambda(lambda x: 2 * x - 1)])
    dataset = MNIST(root=str(dataset_root), train=True, transform=transform, download=True)
    test_dataset = MNIST(root=str(dataset_root), train=False, transform=transform, download=True)
    return dataset, test_dataset


def get_fashion_train_test_datasets(
    dataset_root: Union[str, PathLike],
) -> Tuple[Dataset[tuple[Tensor, LongTensor]], Dataset[tuple[Tensor, LongTensor]]]:
    """
    Fetch Fashion MNIST train and test dataset.

    Parameters
    ----------
    dataset_root: str
        The path where the data is to be downloaded.

    Returns
    -------
    dataset: Dataset
        The training dataset.
    test_dataset: Dataset
        The test dataset.
    """
    transform = Compose([ToTensor(), Lambda(lambda x: 2 * x - 1)])
    dataset = FashionMNIST(root=str(dataset_root), train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(
        root=str(dataset_root), train=False, transform=transform, download=True
    )
    return dataset, test_dataset


def get_emnist_train_test_datasets(
    dataset_root: Union[str, PathLike],
) -> Tuple[Dataset[tuple[Tensor, LongTensor]], Dataset[tuple[Tensor, LongTensor]]]:
    """
    Fetch EMNIST train and test dataset.

    Parameters
    ----------
    dataset_root: str
        The path where the data is to be downloaded.

    Returns
    -------
    dataset: Dataset
        The training dataset.
    test_dataset: Dataset
        The test dataset.
    """
    transform = Compose([ToTensor(), Lambda(lambda x: 2 * x - 1)])
    dataset = EMNIST(
        root=str(dataset_root), split="byclass", train=True, transform=transform, download=True
    )
    test_dataset = EMNIST(
        root=str(dataset_root), split="byclass", train=False, transform=transform, download=True
    )
    return dataset, test_dataset


def get_cifar10_train_test_datasets(
    dataset_root: Union[str, PathLike],
) -> Tuple[Dataset[tuple[Tensor, LongTensor]], Dataset[tuple[Tensor, LongTensor]]]:
    """
    Fetch CIFAR train and test dataset.

    Parameters
    ----------
    dataset_root: str
        The path where the data is to be downloaded.

    Returns
    -------
    dataset: Dataset
        The training dataset.
    test_dataset: Dataset
        The test dataset.
    """
    dataset = CIFAR10(
        root=str(dataset_root), train=True, transform=cifar10_transform_train, download=True
    )
    test_dataset = CIFAR10(
        root=str(dataset_root), train=False, transform=cifar10_transform_test, download=True
    )
    return dataset, test_dataset


def get_cifar100_train_test_datasets(
    dataset_root: Union[str, PathLike],
) -> Tuple[Dataset[tuple[Tensor, LongTensor]], Dataset[tuple[Tensor, LongTensor]]]:
    """
    Fetch MNIST train and test dataset.

    Parameters
    ----------
    dataset_root: str
        The path where the data is to be downloaded.

    Returns
    -------
    dataset: Dataset
        The training dataset.
    test_dataset: Dataset
        The test dataset.
    """
    dataset = CIFAR100(
        root=str(dataset_root), train=True, transform=cifar100_transform_train, download=True
    )
    test_dataset = CIFAR100(
        root=str(dataset_root), train=False, transform=cifar100_transform_test, download=True
    )
    return dataset, test_dataset
