# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.

import copy
import math
import sys
import time
import warnings
from argparse import Namespace
from typing import Tuple

import torch

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.loggers import *  # TODO rigczify
from utils.model_utils import get_num_params, get_num_trainable_params
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    left_bound = k * dataset.args.classes_per_task
    right_bound = (k + 1) * dataset.args.classes_per_task
    outputs[:, :left_bound] = -float("inf")
    outputs[:, right_bound:] = -float("inf")


def evaluate(
    model: ContinualModel, dataset: ContinualDataset, only_last=False
) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    cil_accs, til_accs = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if only_last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == "class-il":
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        cil_accs.append(correct / total * 100)
        til_accs.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return cil_accs, til_accs


def get_reweighted_mean_accs(accs_lists, dataset):
    # Computes average accuracy reweighted by dataset sizes.
    res = []
    for accs in accs_lists:
        assert len(accs) == len(dataset.test_loaders)
        agg_acc, agg_size = 0, 0
        for ta, loader in zip(accs, dataset.test_loaders):
            agg_acc += ta * len(loader.dataset)
            agg_size += len(loader.dataset)
        res.append(agg_acc / agg_size)
    return res


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowandb:
        assert (
            wandb is not None
        ), "Wandb not installed, please install it or run without wandb"
        wandb_config = copy.deepcopy(vars(args))
        wandb_config["num_params_total"] = get_num_params(model)
        wandb_config["num_params_trainable"] = get_num_trainable_params(model)
        wandb_name = getattr(args, "experiment_name", None)

        print("Trainable params: ", wandb_config["num_params_trainable"])

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            entity=args.wandb_entity,
            config=wandb_config,
            anonymous="allow",
            settings=wandb.Settings(start_method="thread"),
        )
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    cil_results, til_results = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=args.verbose)

    print(file=sys.stderr)
    for t in range(dataset.n_tasks):
        model.reset_optimizer()

        model.net.train()
        class_start = dataset.i
        train_loader, _ = dataset.get_data_loaders()
        class_end = dataset.i

        if hasattr(model, "begin_task"):
            model.begin_task(dataset)

        model.reset_per_step_lr_scheduler(
            num_steps=model.args.n_epochs * len(train_loader),
            class_start=class_start,
            class_end=class_end,
        )

        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break

                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss = model.meta_observe(inputs, labels, not_aug_inputs)

                model.step_lr_scheduler()
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

        if hasattr(model, "end_task"):
            model.end_task(dataset)

        if (t + 1) % args.eval_every_n_task == 0 or (t + 1) == dataset.n_tasks:
            accs = evaluate(model, dataset)
            cil_results.append(accs[0])
            til_results.append(accs[1])

            reweighted_mean_acc = get_reweighted_mean_accs(accs, dataset)
            print_mean_accuracy(reweighted_mean_acc, t + 1, dataset.SETTING)

            if not args.disable_log:
                logger.log(reweighted_mean_acc)
                logger.log_fullacc(accs)

                if not args.nowandb:
                    wandb_safe_log(
                        {
                            "cil_mean_acc": reweighted_mean_acc[0],
                            "til_mean_acc": reweighted_mean_acc[1],
                            **{f"cil_acc_task_{i}": a for i, a in enumerate(accs[0])},
                            **{f"til_acc_task_{i}": a for i, a in enumerate(accs[1])},
                        }
                    )

    if not args.disable_log:
        logger.write(vars(args))

    if not args.nowandb:
        wandb.finish()
