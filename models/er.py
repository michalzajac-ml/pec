# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
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
    parser = ArgumentParser(description="ER")
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = "er"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(Er, self).__init__(backbone, loss, dataset_config, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        return loss.item()
