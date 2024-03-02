import torch.nn as nn


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_from_name(name):
    if name == "relu":
        return nn.ReLU
    if name == "lrelu":
        return nn.LeakyReLU
    if name == "leakyrelu":
        return nn.LeakyReLU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    if name == "selu":
        return nn.SELU
    if name == "silu":
        return nn.SiLU
    if name == "celu":
        return nn.CELU
    if name == "mish":
        return nn.Mish
    if name == "no":
        return nn.Identity

    assert False, "bad activation name!"
