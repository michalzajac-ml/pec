from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import ModuleList

from backbone.vae import get_vae
from models.utils.continual_model import ContinualModel, get_lr_scheduler
from utils.args import (
    ArgumentParser,
    add_experiment_args,
    add_management_args,
    str2bool,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="VAE generative classification")
    add_management_args(parser)
    add_experiment_args(parser)

    # -conv-layers
    parser.add_argument(
        "--conv_type", type=str, default="standard", choices=["standard", "resNet"]
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="# blocks per conv-layer (only for 'resNet')",
    )
    parser.add_argument(
        "--depth", type=int, help="# of convolutional layers (0 = only fc-layers)"
    )
    parser.add_argument(
        "--reducing_layers",
        type=int,
        dest="rl",
        help="# of layers with stride (=image-size halved)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=16,
        help="# of channels 1st conv-layer (doubled every 'rl')",
    )
    parser.add_argument(
        "--conv_bn",
        type=str,
        default="yes",
        help="use batch-norm in the conv-layers (yes|no)",
    )
    parser.add_argument(
        "--conv_in",
        type=str,
        default="no",
        help="use instance normalization in the conv-layers (yes|no)",
    )
    parser.add_argument(
        "--conv_ln",
        type=str,
        default="no",
        help="use layer normalization in the conv-layers (yes|no)",
    )
    parser.add_argument(
        "--conv_nl", type=str, default="relu", choices=["relu", "leakyrelu"]
    )
    parser.add_argument(
        "--gp", type=str2bool, default=False, help="ave global pool after conv-layers"
    )
    # -fully-connected-layers
    parser.add_argument(
        "--fc_lay",
        type=int,
        default=3,
        help="# of fully-connected layers",
    )
    parser.add_argument(
        "--fc_units", type=int, metavar="N", help="# of units in first fc-layers"
    )
    parser.add_argument(
        "--fc_drop", type=float, default=0.0, help="dropout probability for fc-units"
    )
    parser.add_argument(
        "--fc_bn",
        type=str,
        default="no",
        help="use batch-norm in the fc-layers (no|yes)",
    )
    parser.add_argument(
        "--fc_ln",
        type=str,
        default="no",
        help="use layer-norm in the fc-layers (no|yes)",
    )
    parser.add_argument(
        "--fc_nl", type=str, default="relu", choices=["relu", "leakyrelu", "none"]
    )
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    parser.add_argument(
        "--z_dim", type=int, default=100, help="size of latent representation (def=100)"
    )
    parser.add_argument(
        "--deconv_type", type=str, default="standard", choices=["standard", "resNet"]
    )
    parser.add_argument(
        "--no_bn_dec", action="store_true", help="don't use batchnorm in decoder"
    )
    parser.add_argument(
        "--prior",
        type=str,
        default="standard",
        choices=["standard", "vampprior", "GMM"],
    )
    parser.add_argument(
        "--n_modes", type=int, default=1, help="how many modes for prior? (def=1)"
    )

    parser.add_argument("--recon_loss", type=str, default="MSE", choices=["MSE", "BCE"])
    parser.add_argument("--importance_samples", type=int, default=100)

    return parser


class VaeGc(ContinualModel):
    NAME = "vaegc"

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        dataset_config: dict,
        args: Namespace,
        transform: nn.Module,
    ) -> None:
        super(VaeGc, self).__init__(backbone, loss, dataset_config, args, transform)

        assert backbone is None

        num_classes = dataset_config["classes"]
        self.vae_modules = ModuleList(
            [get_vae(dataset_config, args) for _ in range(num_classes)]
        )
        self.net = self.vae_modules

        self.opt = [
            self.optim_class(module.parameters(), lr=self.args.lr)
            for module in self.vae_modules
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
        all_scores = []
        for i in range(x.shape[0]):
            x_single = x[i : i + 1]
            scores = []
            for module in self.vae_modules:
                scores.append(
                    module.estimate_loglikelihood_single(
                        x_single, S=self.args.importance_samples
                    )
                )
            all_scores.append(scores)
        return torch.tensor(all_scores, device=self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        losses = []
        for label in set(labels):
            module = self.vae_modules[label]
            opt = self.opt[label]

            module.train()
            if module.convE.frozen:
                module.convE.eval()
            if module.fcE.frozen:
                module.fcE.eval()

            opt.zero_grad()

            x = inputs[labels == label]

            recon_batch, mu, logvar, z = module(x, full=True, reparameterize=True)
            reconL, variatL = module.loss_function(
                x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar
            )
            loss = reconL + variatL

            loss.backward()
            opt.step()

            losses.append(loss.item())

        return torch.mean(torch.tensor(losses)).item()
