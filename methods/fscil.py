"""FSCIL training module."""

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel

from dataloaders.helpter import get_dataloader
from losses.contrastive import SupContrastive
from methods.helper import get_optimizer_base
from models.encoder import FSCILencoder
from utils import dist_utils
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
        self.model: torch.nn.Module = FSCILencoder(args)
        self.criterion = SupContrastive()

        # distributed training setup
        if args.distributed and dist_utils.is_dist_avail_and_initialized():
            device_id = torch.cuda.current_device()
            torch.cuda.set_device(device_id)

            self.model = self.model.cuda(device_id)
            self.criterion = self.criterion.cuda(device_id)

            self.model = DistributedDataParallel(
                self.model,
                device_ids=[device_id],
            )
        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train(self) -> None:
        """Train the model."""
        for session in range(self.args.sessions):
            # train session
            print(f"Training session {session + 1}...")

            # initialize dataset
            train_set, trainloader, testloader = get_dataloader(self.args, session)
            optimizer, scheduler = get_optimizer_base(self.model, self.args)

            for epoch in range(self.args.epochs_base):
                # Unfreeze some encoder parameters at encoder_fine_tuning_start_epoch
                # defined by args.fine_tune_layer_after
                if epoch == self.args.encoder_fine_tuning_start_epoch:
                    status = False
                    for name, param in self.model.encoder_q.named_parameters():
                        if (
                            name.startswith("model.blocks")  # TODO Handle DDP
                            and int(name.split(".")[2])
                            == self.args.fine_tune_layer_after
                        ):
                            status = True
                        param.requires_grad = status
                    for name, param in self.model.encoder_q.named_parameters():
                        print(name, param.requires_grad)

            # distributed sampler

            # dataloaders

            # validate session

            # test session

            # save model
