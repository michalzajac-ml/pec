# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import (
    ArgumentParser,
    add_experiment_args,
    add_management_args,
    add_rehearsal_args,
)
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ER ACE")
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = "er_ace"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(ErACE, self).__init__(backbone, loss, dataset_config, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = dataset_config["classes"]
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max() :] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.0)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
