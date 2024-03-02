# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.

import json
import os
import sys
import time
from contextlib import suppress
from typing import Any, Dict

import numpy as np

import wandb
from utils.conf import base_path_results
from utils.metrics import backward_transfer, forgetting, forward_transfer

useless_args = [
    "dataset",
    "tensorboard",
    "validation",
    "model",
    "csv_log",
    "load_best_args",
]


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int, setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == "domain-il":
        mean_acc, _ = mean_acc
        print(
            "\nAccuracy for {} task(s): {} %".format(task_number, round(mean_acc, 2)),
            file=sys.stderr,
        )
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print(
            "\nAccuracy for {} task(s): \t [Class-IL]: {} %"
            " \t [Task-IL]: {} %\n".format(
                task_number, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2)
            ),
            file=sys.stderr,
        )


def wandb_safe_log(*args, **kwargs):
    #  Try several times.
    for _ in range(10):
        try:
            wandb.log(*args, **kwargs)
        except:
            time.sleep(10)
        else:
            break


class Logger:
    def __init__(self, setting_str: str, dataset_str: str, model_str: str) -> None:
        self.accs = []
        self.fullaccs = []
        if setting_str == "class-il":
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def dump(self):
        dic = {
            "accs": self.accs,
            "fullaccs": self.fullaccs,
            "fwt": self.fwt,
            "bwt": self.bwt,
            "forgetting": self.forgetting,
            "fwt_mask_classes": self.fwt_mask_classes,
            "bwt_mask_classes": self.bwt_mask_classes,
            "forgetting_mask_classes": self.forgetting_mask_classes,
        }
        if self.setting == "class-il":
            dic["accs_mask_classes"] = self.accs_mask_classes
            dic["fullaccs_mask_classes"] = self.fullaccs_mask_classes

        return dic

    def load(self, dic):
        self.accs = dic["accs"]
        self.fullaccs = dic["fullaccs"]
        self.fwt = dic["fwt"]
        self.bwt = dic["bwt"]
        self.forgetting = dic["forgetting"]
        self.fwt_mask_classes = dic["fwt_mask_classes"]
        self.bwt_mask_classes = dic["bwt_mask_classes"]
        self.forgetting_mask_classes = dic["forgetting_mask_classes"]
        if self.setting == "class-il":
            self.accs_mask_classes = dic["accs_mask_classes"]
            self.fullaccs_mask_classes = dic["fullaccs_mask_classes"]

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]

        if self.setting == "class-il":
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == "class-il":
            self.fwt_mask_classes = forward_transfer(
                results_mask_classes, accs_mask_classes
            )

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == "general-continual":
            self.accs.append(mean_acc)
        elif self.setting == "domain-il":
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        if self.setting == "class-il":
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs["cil_mean_step" + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs["cil_task_" + str(j + 1) + "_step_" + str(i + 1)] = acc

        for i, acc in enumerate(self.accs_mask_classes):
            wrargs["til_mean_step" + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs_mask_classes):
            for j, acc in enumerate(fa):
                wrargs["til_task_" + str(j + 1) + "_step_" + str(i + 1)] = acc

        wrargs["forward_transfer"] = self.fwt
        wrargs["backward_transfer"] = self.bwt
        wrargs["forgetting"] = self.forgetting

        log_dir = os.path.join(base_path_results(), self.dataset, self.model)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"seed_{args['seed']}_cpt_{args['classes_per_task']}_{args['conf_jobnum']}.json",
        )

        with open(log_path, "a") as f:
            json.dump(wrargs, f, indent=2)
