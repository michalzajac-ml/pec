# Based on https://github.com/GMvandeVen/class-incremental-learning/blob/main/models/slda.py

import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument(
        "--covariance_type", type=str, choices=["identity", "pure_streaming"]
    )
    parser.add_argument("--slda_epsilon", type=float, default=1e-4)

    return parser


class SLDA(ContinualModel):
    NAME = "slda"

    def __init__(self, backbone, loss, dataset_config, args, transform):
        super(SLDA, self).__init__(backbone, loss, dataset_config, args, transform)

        assert backbone is None
        self.net = torch.nn.Linear(1, 1)  # dummy backbone

        # SLDA parameters
        self.num_features = dataset_config["channels"] * (dataset_config["size"] ** 2)
        self.classes = dataset_config["classes"]
        self.epsilon = args.slda_epsilon
        assert args.covariance_type in ["identity", "pure_streaming"]
        self.covariance_type = args.covariance_type

        # Initialize SLDA class-means & counts
        self.muK = torch.zeros((self.classes, self.num_features)).to(self.device)
        self.cK = torch.zeros(self.classes).to(self.device)
        self.num_updates = 0
        self.new_data = True

        self.Covariance = torch.eye(self.num_features).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.new_data:
            self.new_data = False  # -> only need to convert gen model into classifier again if new data arrives
            # compute the precision-matrix (= inverse of covariance-matrix)
            self.Precision = torch.pinverse(
                (1 - self.epsilon) * self.Covariance
                + self.epsilon * torch.eye(self.num_features).to(self.device)
            )
            # -parameters for linear classifier implied by the generative model:
            M = self.muK.transpose(1, 0)  # -> shape: [num_features]x[classes]
            self.Weights = torch.matmul(
                self.Precision, M
            )  # -> shape: [num_features]x[classes]
            self.biases = 0.5 * torch.sum(
                M * self.Weights, dim=0
            )  # -> shape: [classes]

        # Make predictions for the provided samples, and return them
        with torch.no_grad():
            batch_size = x.shape[0]
            scores = (
                torch.matmul(x.view(batch_size, self.num_features), self.Weights)
                - self.biases
            )  # -> shape: [batch]x[classes]
        return scores

    def observe(self, inputs, labels, not_aug_inputs):
        label = int(labels[0])
        # Technical assumption. Can be guaranteed by one-class tasks or batch_size=1.
        assert torch.all(labels == label)

        inputs = inputs.view(inputs.shape[0], self.num_features)
        x, y = inputs, labels

        with torch.no_grad():
            # Update covariance-matrix (if requested)
            if self.covariance_type == "pure_streaming":
                x_minus_mu = x - self.muK[y]
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Covariance = (self.num_updates * self.Covariance + delta) / (
                    self.num_updates + 1
                )

            # Update class-means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1

        # return fake loss value so that code does not crash.
        return 0.0
