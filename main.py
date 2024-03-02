# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import importlib
import os
import socket
import sys

import numpy  # needed (don't change it)
import torch.nn.functional as F

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

import datetime
import uuid
from argparse import ArgumentParser

import torch

from datasets import get_dataset
from models import get_all_models, get_model
from utils.args import add_management_args
from utils.conf import set_random_seed
from utils.training import train


def parse_args():
    parser = ArgumentParser(description="mammoth", allow_abbrev=False)
    parser.add_argument(
        "--model", type=str, required=True, help="Model name.", choices=get_all_models()
    )
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module("models." + args.model)

    get_parser = getattr(mod, "get_parser")
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)
    dataset_config = {
        "classes": dataset.NUM_CLASSES,
        "size": dataset.IMG_SIZE,
        "channels": dataset.NUM_CHANNELS,
    }

    if args.resize_image_shape > 0:
        assert args.dataset == "seq-miniimg", "Resize only supported for seq-miniimg"
        dataset_config["size"] = args.resize_image_shape

    if args.balance_truncate_data:
        assert (
            args.dataset == "seq-svhn"
        ), "Balance truncate only supported for seq-svhn"

    if (
        hasattr(importlib.import_module("models." + args.model), "Buffer")
        and args.minibatch_size is None
    ):
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = None
    loss = None
    if args.model not in [
        "pec",
        "vaegc",
        "slda",
    ]:  # For those, backbone is defined in the model
        backbone = dataset.get_backbone()
        loss = F.cross_entropy
    model = get_model(dataset_config, args, backbone, loss, dataset.get_transform())

    if args.debug_mode:
        args.nowandb = True

    train(model, dataset, args)


if __name__ == "__main__":
    main()
