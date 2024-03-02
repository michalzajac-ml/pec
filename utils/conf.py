# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path_results() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return "./results/"


def base_path_dataset() -> str:
    # TODO: clean up
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    if "athena" in os.environ.get("SLURM_SUBMIT_HOST", ""):
        return "/net/pr2/projects/plgrid/plgggmum_crl/mzajac/torch_datasets/"
    return os.path.expanduser("~/.torch_datasets/")


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
