# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

from typing import Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from backbone.pec_modules import PecCNN
from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(
            root,
            train,
            transform,
            target_transform,
            download=not self._check_integrity(),
        )


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(
            root,
            train,
            transform,
            target_transform,
            download=not self._check_integrity(),
        )

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
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

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):
    NAME = "seq-cifar10"
    SETTING = "class-il"
    NUM_CLASSES = 10
    IMG_SIZE = 32
    NUM_CHANNELS = 3
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()]
        )
        if self.args.force_no_augmentations:
            transform = test_transform

        train_dataset = MyCIFAR10(
            base_path() + "CIFAR10",
            train=True,
            download=True,
            transform=transform,
        )
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, test_transform, self.NAME
            )
        else:
            test_dataset = TCIFAR10(
                base_path() + "CIFAR10",
                train=False,
                download=True,
                transform=test_transform,
            )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM]
        )
        return transform

    def get_backbone(self):
        num_classes = SequentialCIFAR10.NUM_CLASSES
        backbone = self.args.backbone or "resnet"

        if backbone == "resnet":
            return resnet18(num_classes, [2, 2, 2, 2], 64)

        if backbone == "pec_cnn":
            dataset_config = {"channels": 3, "size": 32}
            return PecCNN(dataset_config, self.args)

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()
