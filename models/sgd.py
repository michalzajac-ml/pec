# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.


from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Vanilla SGD (fine-tuning)")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = "sgd"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(Sgd, self).__init__(backbone, loss, dataset_config, args, transform)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
