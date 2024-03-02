# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.


import importlib
import os


def get_all_models():
    return [
        model.split(".")[0]
        for model in os.listdir("models")
        if not model.find("__") > -1 and "py" in model
    ]


names = {}
for model in get_all_models():
    mod = importlib.import_module("models." + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace("_", "")]
    names[model] = getattr(mod, class_name)


def get_model(dataset_config, args, backbone, loss, transform):
    return names[args.model](backbone, loss, dataset_config, args, transform)
