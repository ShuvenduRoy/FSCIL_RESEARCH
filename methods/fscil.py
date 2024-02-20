"""FSCIL training module."""

import argparse

from models.encoder import FSCILencoder
from utils.dist_utils import is_main_process
from utils.train_utils import ensure_path


class FSCILTrainer:
    """FSCIL Trainer class."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize FSCIL Trainer.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Arguments passed to the trainer.

        Returns
        -------
        None
        """
        self.args = args

        # train statistics
        self.trlog = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_acc": [],
            "base_acc": [],
            "val_acc": [],
            "test_acc": [],
            "max_acc_epoch": 0,
            "max_acc": [0.0] * args.sessions,
            "max_base_acc": [0.0] * args.sessions,
        }
        if is_main_process():
            ensure_path(args.save_path)

        # initialize model
        self.model = FSCILencoder(args)

        # initialize dataset

        # distributed sampler

        # dataloaders

    def train(self) -> None:
        """Train the model."""
        pass
