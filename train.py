"""The main training file."""

import argparse
import os
from pprint import pprint

from methods.fscil import FSCILTrainer
from utils.train_utils import get_command_line_parser, get_dataset_configs, set_seed


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
    trainer = FSCILTrainer(args)

    trainer.train()


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = get_dataset_configs(args)

    # set the seed
    set_seed(args.seed)
    pprint(vars(args))
    main(args)
