from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class LabelsTrick(ContinualModel):
    NAME = "labels_trick"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(LabelsTrick, self).__init__(
            backbone, loss, dataset_config, args, transform
        )
        self.class_start, self.class_end = -1, 0

    def begin_task(self, dataset):
        self.class_start = self.class_end
        self.class_end = dataset.i

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)[:, self.class_start : self.class_end]
        labels -= self.class_start
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
