import torch
import torch.nn as nn
import torchvision

from utils.model_utils import get_activation_from_name


def get_single_pec_network(dataset_config, args, is_teacher=False):
    model_class = {"mlp": PecMLP, "cnn": PecCNN}[args.pec_architecture]
    model = model_class(dataset_config, args, is_teacher=is_teacher)

    if is_teacher:
        for param in model.parameters():
            param.requires_grad = False

    return model


class PecMLP(nn.Module):
    def __init__(self, dataset_config, args, is_teacher=False) -> None:
        super().__init__()

        num_layers = args.pec_num_layers

        width = args.pec_width
        if is_teacher:
            width = int(width * args.pec_teacher_width_multiplier)

        in_channels = (dataset_config["size"] ** 2) * dataset_config["channels"]
        hidden_channels = num_layers * [width]
        hidden_channels[-1] = args.pec_output_dim

        norm_layer = nn.LayerNorm if args.pec_normalize_layers else None

        self.features = nn.Sequential(
            nn.Flatten(),
            torchvision.ops.MLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                norm_layer=norm_layer,
                activation_layer=get_activation_from_name(args.pec_activation),
                inplace=None,
                dropout=0.0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PecCNN(nn.Module):
    def __init__(self, dataset_config, args, is_teacher=False) -> None:
        super().__init__()

        layers = []

        cur_channels = dataset_config["channels"]
        cur_size = dataset_config["size"]
        for channels, kernel, stride in args.pec_conv_layers:
            if is_teacher:
                channels = int(args.pec_teacher_width_multiplier * channels)

            layers.append(
                nn.Conv2d(
                    in_channels=cur_channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                )
            )
            cur_channels = channels
            cur_size //= stride

            if args.pec_normalize_layers:
                layers.append(nn.InstanceNorm2d(cur_channels, affine=True))

            layers.append(get_activation_from_name(args.pec_activation)())

        layers.append(nn.AdaptiveAvgPool2d(args.pec_conv_reduce_spatial_to))
        layers.append(nn.Flatten(1))
        layers.append(
            nn.Linear(
                cur_channels * (args.pec_conv_reduce_spatial_to**2),
                args.pec_output_dim,
            )
        )

        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
