# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.


import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class ValidationDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: np.ndarray,
        transform: Optional[nn.Module] = None,
        target_transform: Optional[nn.Module] = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, np.ndarray):
            if np.max(img) < 2:
                img = Image.fromarray(np.uint8(img * 255))
            else:
                img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_train_val(
    train: Dataset, test_transform: nn.Module, dataset: str, val_perc: float = 0.1
):
    """
    Extract val_perc% of the training set as the validation set.
    :param train: training dataset
    :param test_transform: transformation of the test dataset
    :param dataset: dataset name
    :param val_perc: percentage of the training set to be extracted
    :return: the training set and the validation set
    """
    dataset_length = train.data.shape[0]
    # For now, disable caching.
    # directory = 'datasets/val_permutations/'
    # os.makedirs(directory, exist_ok=True)
    # file_name = dataset + '.pt'
    # if os.path.exists(directory + file_name):
    #     perm = torch.load(directory + file_name)
    # else:
    #     perm = torch.randperm(dataset_length)
    #     torch.save(perm, directory + file_name)
    perm = torch.randperm(dataset_length)
    train.data = train.data[perm]
    train.targets = np.array(train.targets)[perm]
    test_dataset = ValidationDataset(
        train.data[: int(val_perc * dataset_length)],
        train.targets[: int(val_perc * dataset_length)],
        transform=test_transform,
    )
    train.data = train.data[int(val_perc * dataset_length) :]
    train.targets = train.targets[int(val_perc * dataset_length) :]

    return train, test_dataset
