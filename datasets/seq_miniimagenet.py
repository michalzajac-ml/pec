# Based on code from learn2learn library.

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from backbone.ResNet18 import resnet18
from datasets import ContinualDataset
from datasets.transforms.denormalization import DeNormalize
from datasets.utils import download_file
from datasets.utils.continual_dataset import store_masked_loaders
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path


def index_classes(items, shift=0):
    idx = {}
    for i in items:
        if i not in idx:
            idx[i] = len(idx) + shift
    return idx


DOWNLOAD_FILE_PATHS = {
    "train": "https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1",
    "validation": "https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1",
    "test": "https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1",
}


class MiniImagenet(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py)
    **Description**
    The *mini*-ImageNet dataset was originally introduced by Vinyals et al., 2016.
    It consists of 60'000 colour images of sizes 84x84 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the ImageNet dataset, and we use the splits from Ravi & Larochelle, 2017.
    **References**
    1. Vinyals et al. 2016. “Matching Networks for One Shot Learning.” NeurIPS.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.
    **Arguments**
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Download the dataset if it's not available.
    **Example**
    ~~~python
    train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        return_not_aug=False,
    ):
        super(MiniImagenet, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.return_not_aug = return_not_aug

        for mode in ["train", "validation", "test"]:
            data_path = os.path.join(self.root, "mini-imagenet-cache-" + mode + ".pkl")
            if not os.path.exists(data_path):
                print(f"Downloading {data_path}...")
                download_file(DOWNLOAD_FILE_PATHS[mode], data_path)

        # Merge train, valid, test as they correspond to different classes (for few-shot learning).
        # Later we will create our own train/test division.
        pickle_files = [
            os.path.join(self.root, "mini-imagenet-cache-" + mode + ".pkl")
            for mode in ["train", "validation", "test"]
        ]
        data = []
        for pickle_file in pickle_files:
            with open(pickle_file, "rb") as f:
                data.append(pickle.load(f))

        all_x, all_y = [], []
        class_shift = 0
        for d in data:
            x = d["image_data"]
            y = np.ones(len(x), dtype=np.int64)

            class_idx = index_classes(sorted(d["class_dict"].keys()), shift=class_shift)
            for class_name, idxs in d["class_dict"].items():
                for idx in idxs:
                    y[idx] = class_idx[class_name]

            all_x.append(x)
            all_y.append(y)
            class_shift += len(np.unique(y))
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        # Create train/test division.
        NUM_CLASSES = 100
        ALL_PER_CLASS = 600
        TRAIN_PER_CLASS = 500

        self.data, self.targets = [], []
        for c in range(NUM_CLASSES):
            ind = all_y == c
            assert sum(ind) == ALL_PER_CLASS
            x, y = all_x[ind], all_y[ind]
            if train:
                x, y = x[:TRAIN_PER_CLASS], y[:TRAIN_PER_CLASS]
            else:
                x, y = x[TRAIN_PER_CLASS:], y[TRAIN_PER_CLASS:]
            self.data.append(x)
            self.targets.append(y)
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

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

    def __len__(self):
        return len(self.data)


class SequentialMiniImageNet(ContinualDataset):
    NAME = "seq-miniimg"
    SETTING = "class-il"
    NUM_CLASSES = 100
    IMG_SIZE = 84
    NUM_CHANNELS = 3
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(84, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4729, 0.4487, 0.4030), (0.2833, 0.2752, 0.2886)),
        ]
    )

    def get_data_loaders(self):
        transform = self.TRANSFORM
        if self.args.resize_image_shape > 0:
            transform = transforms.Compose(
                [
                    transforms.Resize(self.args.resize_image_shape),
                    transforms.RandomCrop(self.args.resize_image_shape, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4729, 0.4487, 0.4030), (0.2833, 0.2752, 0.2886)
                    ),
                ]
            )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()]
        )
        if self.args.resize_image_shape > 0:
            test_transform = transforms.Compose(
                [
                    transforms.Resize(self.args.resize_image_shape),
                    transforms.ToTensor(),
                    self.get_normalization_transform(),
                ]
            )
        if self.args.force_no_augmentations:
            transform = test_transform

        train_dataset = MiniImagenet(
            base_path(),
            train=True,
            transform=transform,
            return_not_aug=True,
        )
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(
                train_dataset, test_transform, self.NAME
            )
        else:
            test_dataset = MiniImagenet(
                base_path(), train=False, transform=test_transform
            )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialMiniImageNet.TRANSFORM]
        )
        return transform

    def get_backbone(self):
        num_classes = SequentialMiniImageNet.NUM_CLASSES
        return resnet18(num_classes, [2, 2, 2, 2], 64)

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            (0.4729, 0.4487, 0.4030), (0.2833, 0.2752, 0.2886)
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4729, 0.4487, 0.4030), (0.2833, 0.2752, 0.2886))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialMiniImageNet.get_batch_size()
