from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import SVHN

from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class MySVHN(SVHN):
    """
    Overrides the MNIST dataset to change the getitem function.
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        return_not_aug=False,
        balance_truncate_data=False,
    ) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MySVHN, self).__init__(root, split, transform, target_transform, download)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.targets = self.labels
        self.return_not_aug = return_not_aug

        if balance_truncate_data:
            min_size = min([sum(self.targets == c) for c in range(10)])
            indices = []
            for c in range(10):
                ind_c = np.nonzero(self.targets == c)[0][:min_size]
                indices.append(ind_c)
            indices = np.concatenate(indices)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # to return a PIL Image
        img = Image.fromarray(img, mode="RGB")
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_not_aug:
            return img, target, not_aug_img
        else:
            return img, target


class SequentialSVHN(ContinualDataset):
    NAME = "seq-svhn"
    SETTING = "class-il"
    NUM_CLASSES = 10
    IMG_SIZE = 32
    NUM_CHANNELS = 3
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()]
        )
        if self.args.force_no_augmentations:
            transform = test_transform

        train_dataset = MySVHN(
            base_path() + "SVHN",
            split="train",
            download=True,
            transform=transform,
            return_not_aug=True,
            balance_truncate_data=self.args.balance_truncate_data,
        )

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, test_transform, self.NAME
            )
        else:
            test_dataset = MySVHN(
                base_path() + "SVHN",
                split="test",
                download=True,
                transform=test_transform,
                balance_truncate_data=self.args.balance_truncate_data,
            )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialSVHN.TRANSFORM]
        )
        return transform

    def get_backbone(self):
        num_classes = SequentialSVHN.NUM_CLASSES
        return resnet18(num_classes, [2, 2, 2, 2], 64)

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialSVHN.get_batch_size()
