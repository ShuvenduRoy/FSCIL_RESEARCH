"""FSCIL training module."""

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel

from dataloaders.helpter import get_dataloader
from losses.contrastive import SupContrastive
from methods.helper import get_optimizer_base, replace_base_fc, test, train_one_epoch
from models.encoder import FSCILencoder
from utils import dist_utils
from utils.dist_utils import is_main_process
from utils.train_utils import ensure_path


class FSCITTrainer:
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
        self.criterion = SupContrastive()
        self.optimizer, self.scheduler = get_optimizer_base(self.model, self.args)
        self.device_id = None

        # distributed training setup
        self.model_without_ddp = self.model
        if args.distributed and dist_utils.is_dist_avail_and_initialized():
            self.device_id = torch.cuda.current_device()
            torch.cuda.set_device(self.device_id)

            self.model = self.model.cuda(self.device_id)
            self.criterion = self.criterion.cuda(self.device_id)

            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device_id],
            )
            self.model_without_ddp = self.model.module

        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def adjust_learnable_parameters(self, session: int, epoch: int = 0) -> None:
        """Adjust the learnable parameters base of config and current step.

        Parameters
        ----------
        session: int
            Current session
        epoch: int
            Current epoch
        """
        # handle trainable parameters in the base session
        # at certain epochs
        if session == 0:
            # Unfreeze some encoder parameter at encoder_ft_start_epoch
            # defined by args.encoder_ft_start_layer
            if (
                epoch == self.args.encoder_ft_start_epoch
                and self.args.encoder_ft_start_layer != -1
            ):
                status = False
                for (
                    name,
                    param,
                ) in self.model_without_ddp.encoder_q.named_parameters():
                    if (
                        name.startswith("model.blocks")
                        and int(name.split(".")[2]) == self.args.encoder_ft_start_layer
                    ):
                        status = True
                    param.requires_grad = status
                for (
                    name,
                    param,
                ) in self.model_without_ddp.encoder_q.named_parameters():
                    print(
                        "ecnoder_q @session {} @epoch {},".format(session, epoch),
                        name,
                        param.requires_grad,
                    )

        # handle trainable parameters for incremental sessions
        else:
            # Freeze the encoder
            for _, param in self.model_without_ddp.encoder_q.named_parameters():
                param.requires_grad = False
            # Tune params as defined in config # TODO handle what to tune in the inc

            for (
                name,
                param,
            ) in self.model_without_ddp.encoder_q.named_parameters():
                print(
                    "ecnoder_q @session {} @epoch {},".format(session, epoch),
                    name,
                    param.requires_grad,
                )

    def train(self) -> None:
        """Train the model."""
        session_accuracies = {
            "base": [0] * self.args.sessions,
            "incremental": [0] * self.args.sessions,
            "all": [0] * self.args.sessions,
        }
        for session in range(self.args.sessions):
            # initialize dataset
            train_set, trainloader, testloader = get_dataloader(self.args, session)

            # train session
            print(f"Training session {session + 1}...")
            print(f"Train set size: {len(train_set)}")
            print(f"Test set size: {len(testloader.dataset)}")

            if session == 0:  # base session
                # Attempt auto resume # TODO
                for epoch in range(self.args.epochs_base):
                    if dist_utils.is_dist_avail_and_initialized():
                        trainloader.sampler.set_epoch(epoch)
                        testloader.sampler.set_epoch(epoch)

                    # adjust learnable params
                    self.adjust_learnable_parameters(session)

                    # train and test
                    train_one_epoch(
                        model=self.model,
                        trainloader=trainloader,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        args=self.args,
                        device_id=self.device_id,
                    )
                    self.scheduler.step()
                    base_acc, inc_acc, all_acc = test(
                        model=self.model,
                        testloader=testloader,
                        epoch=epoch,
                        args=self.args,
                        session=session,
                        device_id=self.device_id,
                    )
                    # TODO hold the bast model in a variable

                    # TODO save everything for auto resume capability
                    session_accuracies["base"][session] = max(
                        base_acc,
                        session_accuracies["base"][session],
                    )
                    session_accuracies["incremental"][session] = max(
                        inc_acc,
                        session_accuracies["incremental"][session],
                    )
                    session_accuracies["all"][session] = max(
                        all_acc,
                        session_accuracies["all"][session],
                    )
            if self.args.update_base_classifier_with_prototypes:
                # replace base classifier weight with prototypes
                replace_base_fc(train_set, self.model, self.args, self.device_id)
                test(
                    model=self.model,
                    testloader=testloader,
                    epoch=epoch,
                    args=self.args,
                    session=session,
                    device_id=self.device_id,
                )

            else:
                replace_base_fc(train_set, self.model, self.args, self.device_id)
                test(
                    model=self.model,
                    testloader=testloader,
                    epoch=epoch,
                    args=self.args,
                    session=session,
                    device_id=self.device_id,
                )

            # save model # TODO
