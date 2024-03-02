# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import copy

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Learning without Forgetting + Labels Trick")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--alpha", type=float, default=0.5, help="Penalty weight.")
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=2,
        help="Temperature of the softmax function.",
    )
    return parser


class LwfLT(ContinualModel):
    NAME = "lwf_lt"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(LwfLT, self).__init__(backbone, loss, dataset_config, args, transform)
        self.old_net = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.current_task = 0

        self.class_start, self.class_end = -1, 0

    def begin_task(self, dataset):
        self.class_start = self.class_end
        self.class_end = dataset.i

    def end_task(self, dataset):
        self.current_task += 1

        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        for param in self.old_net.parameters():
            param.requires_grad = False

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)

        loss = self.loss(
            outputs[:, self.class_start : self.class_end], labels - self.class_start
        )

        if self.current_task > 0:
            with torch.no_grad():
                old_outputs = self.old_net(inputs)
                old_dist = F.softmax(
                    old_outputs[:, : self.class_start] / self.args.softmax_temp, dim=1
                )
            new_logits = outputs[:, : self.class_start] / self.args.softmax_temp
            loss += self.args.alpha * self.loss(new_logits, old_dist)

        loss.backward()
        self.opt.step()

        return loss.item()
