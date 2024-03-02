# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import argparse
from argparse import ArgumentParser
from typing import Union

from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--resize_image_shape",
        type=int,
        default=-1,
        help="If positive, resize images to this shape. Available only for miniImageNet dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Class-incremental learning method name.",
        choices=get_all_models(),
    )
    parser.add_argument("--classes_per_task", type=int)

    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument("--force_no_augmentations", type=str2bool, default=False)
    parser.add_argument(
        "--optim_kind", type=str, default="adam", choices=["sgd", "adam"]
    )
    parser.add_argument(
        "--optim_wd", type=float, default=0.0, help="optimizer weight decay."
    )
    parser.add_argument(
        "--optim_mom", type=float, default=0.0, help="optimizer momentum."
    )
    parser.add_argument(
        "--optim_nesterov", type=int, default=0, help="optimizer nesterov momentum."
    )
    parser.add_argument(
        "--optim_scheduler", type=str, default="none", choices=["none", "linear"]
    )

    parser.add_argument("--n_epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size.")

    parser.add_argument(
        "--balance_truncate_data",
        type=str2bool,
        default=False,
        help="If true, truncate the dataset to have the same number of samples per class. Available only for SVHN dataset.",
    )

    parser.add_argument("--backbone", type=str)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="The random seed.")

    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Make progress bars verbose",
    )
    parser.add_argument("--disable_log", type=str2bool, help="Disable logging")

    parser.add_argument(
        "--validation",
        type=str2bool,
        help="Test on the validation set",
    )
    parser.add_argument(
        "--debug_mode",
        type=str2bool,
        help="Run only a few forward steps per epoch",
    )
    parser.add_argument("--nowandb", type=str2bool, help="Inhibit wandb logging")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--eval_every_n_task", type=int, default=1)


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument(
        "--buffer_size", type=int, required=True, help="The size of the memory buffer."
    )
    parser.add_argument(
        "--minibatch_size", type=int, help="The batch size of the memory buffer."
    )


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
