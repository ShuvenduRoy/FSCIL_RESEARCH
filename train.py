"""The main training file."""

import argparse
import os
from pprint import pprint

from methods.fscil import FSCILTrainer
from utils import dist_utils
from utils.train_utils import (
    get_command_line_parser,
    get_dataset_configs,
    override_training_configs,
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
    if args.distributed:
        dist_utils.init_distributed_mode(
            launcher=args.distributed_launcher,
            backend=args.distributed_backend,
        )

    trainer = FSCILTrainer(args)

    trainer.train()


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = get_dataset_configs(args)
    args = override_training_configs(args)

    # set the seed
    pprint(vars(args))
    main(args)
