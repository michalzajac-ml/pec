# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.


import sys
from argparse import Namespace

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

from utils.conf import get_device
from utils.loggers import wandb_safe_log
from utils.magic import persistent_locals
from utils.schedulers import PolynomialLR


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """

    NAME: str

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        dataset_config: dict,
        args: Namespace,
        transform: nn.Module,
    ) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.dataset_config = dataset_config
        self.args = args
        self.transform = transform
        self.optim_class = {"sgd": SGD, "adam": Adam}[args.optim_kind]
        self.opt = None
        self.per_step_lr_scheduler = None
        self.device = get_device()

        if not self.NAME:
            raise NotImplementedError("Please specify the name of the model.")

    def reset_optimizer(self):
        self.opt = self.optim_class(self.net.parameters(), lr=self.args.lr)

    def reset_per_step_lr_scheduler(self, num_steps, class_start, class_end):
        self.per_step_lr_scheduler = get_lr_scheduler(
            self.opt, self.args, num_steps=num_steps
        )

    def step_lr_scheduler(self):
        self.per_step_lr_scheduler.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if "wandb" in sys.modules and not self.args.nowandb:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(
        self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor
    ) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowandb and not self.args.debug_mode:
            wandb_safe_log(
                {
                    k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                    for k, v in locals.items()
                    if k.startswith("_wandb_") or k.startswith("loss")
                }
            )


def get_lr_scheduler(optimizer, args, num_steps):
    if args.optim_scheduler == "none":
        return ExponentialLR(optimizer, gamma=1.0)
    if args.optim_scheduler == "linear":
        return PolynomialLR(optimizer, total_iters=num_steps)
    assert False, "Bad optim_scheduler!"
