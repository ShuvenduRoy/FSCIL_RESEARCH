"""The main training file."""

import argparse
import os
from pprint import pprint

import numpy as np

from methods.fscit import FSCITTrainer
from utils.train_utils import (
    get_command_line_parser,
    get_dataset_configs,
    override_training_configs,
    sanity_check,
    save_results,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args: argparse.Namespace) -> None:
    """Train the model.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    None
    """
    results_dict: dict = {}
    # Loop over seeds
    for i in range(args.num_seeds):
        print(f"!!!Training with seed {i}")
        args.seed = i
        trainer = FSCITTrainer(args)
        trainer.train()

        # update the results dictionary
        for key, val in trainer.session_accuracies.items():
            results_dict[key] = results_dict.get(key, []) + [val]

    # averae across seeds
    results_dict["last" + "_std"] = np.array(results_dict["last"]).std(axis=0)
    results_dict["last" + "_std"] = [round(v, 2) for v in results_dict["last" + "_std"]]
    for key in results_dict:
        results_dict[key] = np.array(results_dict[key]).mean(axis=0)
        results_dict[key] = [round(v, 2) for v in results_dict[key]]

    # print the results
    print("Final Results:")
    for key, val in results_dict.items():
        print(f"{key}: {val}")
    save_results(results_dict, args)


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    pprint(vars(args))
    main(args)
