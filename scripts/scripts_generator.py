"""Generates experiment script."""

import argparse
import copy
import importlib
import os
import random
from itertools import product


def generate_scirpts(args: argparse.Namespace) -> None:
    """Generate scripts for experiments."""
    # load the run.sh file
    with open("run.sh", "r") as f:
        original_run = f.readlines()
    original_run.append("\n\n")
    # load the config file
    path = "scripts/configs/{}.yaml".format(args.config)
    yaml = importlib.import_module("yaml")

    with open(path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    configs = configs["parameters"]

    # loop over combinations of hyperparameters
    train_args = ""
    sweep_params = []
    sweep_values = []
    for key, val in configs.items():
        if "value" in val:
            train_args += " --{} {}".format(key, val["value"])
        else:
            sweep_params.append(key)
            sweep_values.append(val["values"])

    # generate all combinations of hyperparameters
    sweep = list(product(*sweep_values))
    random.shuffle(sweep)

    run = copy.deepcopy(original_run)
    # loop over all combinations
    for i in range(len(sweep)):
        # generate the command
        command = ""
        for j in range(len(sweep_params)):
            command += " --{} {}".format(sweep_params[j], sweep[i][j])
        print("python -W ignore train.py " + train_args + command)
        command = 'COMMAND="python -W ignore train.py ' + train_args + command + '"\n'

        if i % args.jobs_per_run == 0 or i == len(sweep) - 1:
            run[20] = "cd ../..\n"
            run[21] = command
            run[7] = (
                "#SBATCH --error=/scratch/a/amiilab/shuvendu/OUTPUTS/FSCIT/{}_exp{}.out\n".format(
                    args.config,
                    args.exp_counter,
                )
            )
            run[8] = (
                "#SBATCH --output=/scratch/a/amiilab/shuvendu/OUTPUTS/FSCIT/{}_exp{}.out\n".format(
                    args.config,
                    args.exp_counter,
                )
            )
            run[11] = "#SBATCH --job-name=exp_{}\n".format(args.exp_counter)

            save_path = "scripts/bash/"
            os.makedirs(save_path, exist_ok=True)

            with open(save_path + "/exp{}.sh".format(args.exp_counter), "w") as f:
                f.writelines(run)
            args.exp_counter += 1
            run = copy.deepcopy(original_run)
        else:
            run.append(command)
            run.append('echo "$COMMAND"\n$COMMAND\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment scripts.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="The config file to use.",
    )
    parser.add_argument(
        "--exp_counter",
        "-e",
        type=int,
        default=1000,
        help="The experiment counter.",
    )
    parser.add_argument(
        "--jobs_per_run",
        "-j",
        type=int,
        default=1,
        help="The number of jobs per run.",
    )
    args = parser.parse_args()
    generate_scirpts(args)
