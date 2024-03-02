# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.


from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import (
    ArgumentParser,
    add_experiment_args,
    add_management_args,
    add_rehearsal_args,
)
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Continual learning via Dark Experience Replay."
    )
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--alpha", type=float, required=False, help="Penalty weight.")
    return parser


class Der(ContinualModel):
    NAME = "der"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(Der, self).__init__(backbone, loss, dataset_config, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
