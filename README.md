# Prediction Error-based Classification (PEC)

This is the official implementation of the method and main experiments from the paper
["Prediction Error-based Classification for Class-Incremental Learning"](https://arxiv.org/abs/2305.18806)
by Michał Zając, Tinne Tuytelaars, and Gido M. van de Ven.

The codebase is heavily based on the [mammoth](https://github.com/aimagelab/mammoth) repository. Some of the code is also based on the [class-incremental-learning](https://github.com/GMvandeVen/class-incremental-learning) repository.

## Preparation

The code uses Python 3, PyTorch, and packages listed in `requirements.txt`. If you use pip, you can install them with `pip install -r requirements.txt`. We include the exact package versions that we tested to work in `requirements_exact_versions.txt`, although these exact versions are not required.

## Running

### Example run
In order to run a single experiment, you need to use `main.py` script, and specify arguments such as dataset, method name, etc. General arguments are defined in `utils/args.py` and method-specific arguments are defined in a file in `models/` corresponding to a given method.

Example MNIST run for PEC method:
```
python3 main.py --dataset=seq-mnist --model=pec --n_epochs=1 \
        --classes_per_task=1 --optim_scheduler=linear --eval_every_n_task=10 \
        --force_no_augmentations=True --batch_size=1 --lr=0.01 \
        --pec_architecture=mlp --pec_activation=gelu \
        --pec_teacher_width_multiplier=500 --pec_width=10 --pec_output_dim=99
```

After running, logs will be saved to a file in `results/` directory, as well as to a wandb run (a link will be displayed).

### Reproducing main results from the paper
We prepared bash scripts to reproduce results from Table 1 and Table 2 under the `scripts/` directory. Each script corresponds to a single result from one of the tables. For example, to reproduce PEC results on CIFAR-10 from Table 1, run 

```
scripts/table1/cifar10/pec.sh 10
```

The only argument in the script is the number of seeds to run (we used 10 in the paper).



## Citation
If you found our codebase useful, please consider citing our paper:

    @inproceedings{
        zajac2024prediction,
        title={Prediction Error-based Classification for Class-Incremental Learning},
        author={Micha{\l} Zaj{\k{a}}c and Tinne Tuytelaars and Gido M van de Ven},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=DJZDgMOLXQ}
    }
