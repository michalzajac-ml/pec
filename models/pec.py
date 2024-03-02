from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import ModuleList

from backbone.pec_modules import get_single_pec_network
from models.utils.continual_model import ContinualModel, get_lr_scheduler
from utils.args import (
    ArgumentParser,
    add_experiment_args,
    add_management_args,
    str2bool,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prediction Error-based Classification")
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument("--pec_architecture", type=str, choices=["mlp", "cnn"])
    parser.add_argument("--pec_num_layers", type=int, default=2)
    parser.add_argument("--pec_width", type=int)
    parser.add_argument("--pec_teacher_width_multiplier", type=int, default=100)
    parser.add_argument("--pec_output_dim", type=int)
    parser.add_argument("--pec_activation", type=str, default="relu")
    parser.add_argument("--pec_normalize_layers", type=str2bool, default=True)
    parser.add_argument("--pec_conv_layers", nargs="+", type=eval)
    parser.add_argument("--pec_conv_reduce_spatial_to", type=int)

    return parser


class Pec(ContinualModel):
    NAME = "pec"

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        dataset_config: dict,
        args: Namespace,
        transform: nn.Module,
    ) -> None:
        super(Pec, self).__init__(backbone, loss, dataset_config, args, transform)

        assert backbone is None

        num_classes = dataset_config["classes"]
        teacher = get_single_pec_network(dataset_config, args, is_teacher=True)
        self.pec_modules = ModuleList(
            [
                PecStudentTeacherPair(dataset_config, args, teacher=teacher)
                for _ in range(num_classes)
            ]
        )
        for i in range(1, num_classes):
            self.pec_modules[i].student.load_state_dict(
                self.pec_modules[0].student.state_dict()
            )
        self.net = self.pec_modules

        self.opt = [
            self.optim_class(module.student.parameters(), lr=self.args.lr)
            for module in self.pec_modules
        ]
        self.per_step_lr_scheduler = [None for _ in range(num_classes)]

        self.cur_class_start = None
        self.cur_class_end = None

    def reset_optimizer(self):
        pass

    def reset_per_step_lr_scheduler(self, num_steps, class_start, class_end):
        self.cur_class_start = class_start
        self.cur_class_end = class_end

        for c in range(class_start, class_end):
            self.per_step_lr_scheduler[c] = get_lr_scheduler(
                self.opt[c], self.args, num_steps=num_steps
            )

    def step_lr_scheduler(self):
        for c in range(self.cur_class_start, self.cur_class_end):
            self.per_step_lr_scheduler[c].step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = []
        with torch.no_grad():
            for module in self.pec_modules:
                scores.append(module(x))
        scores = -torch.stack(scores, dim=1)
        return scores

    def observe(self, inputs, labels, not_aug_inputs):
        loss_acc = 0.0

        for c in range(self.cur_class_start, self.cur_class_end):
            c_inputs = inputs[labels == c]
            if len(c_inputs) == 0:
                continue

            module = self.pec_modules[c]
            opt = self.opt[c]

            opt.zero_grad()
            loss = torch.mean(module(c_inputs))
            loss.backward()
            opt.step()

            loss_acc += float(loss) * c_inputs.shape[0]

        return loss_acc / inputs.shape[0]


class PecStudentTeacherPair(nn.Module):
    def __init__(self, dataset_config, args, teacher):
        super().__init__()
        self.student = get_single_pec_network(dataset_config, args)
        self.teacher = teacher

    def forward(self, x):
        return torch.mean((self.student(x) - self.teacher(x).detach()) ** 2, dim=1)
