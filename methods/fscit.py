"""FSCIL training module."""

import argparse
from copy import deepcopy
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel

from dataloaders.helpter import get_dataloader
from losses.contrastive import SupContrastive
from methods.helper import (
    get_optimizer_base,
    replace_fc_with_prototypes,
    test,
    train_one_epoch,
)
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

        # train statistics. # TODO:CLEAN: probably never used
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
        self.best_model_dict = deepcopy(self.model_without_ddp.state_dict())
        if args.distributed and dist_utils.is_dist_avail_and_initialized():
            self.device_id = torch.cuda.current_device()
            torch.cuda.set_device(self.device_id)

            self.model = self.model.cuda(self.device_id)
            self.criterion = self.criterion.cuda(self.device_id)

            self.model = DistributedDataParallel(  # type: ignore
                self.model,
                device_ids=[self.device_id],
            )
            self.model_without_ddp = self.model.module  # type: ignore

        elif torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def adjust_learnable_parameters(self, session: int, epoch: int) -> None:
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
            if (
                epoch == self.args.encoder_ft_start_epoch
            ):  # current epoch is ft start epoch

                status = self.args.encoder_ft_start_layer == -1  # full fine-tune
                for (
                    name,
                    param,
                ) in self.model_without_ddp.encoder_q.named_parameters():
                    if (
                        name.startswith("model.blocks")
                        and int(name.split(".")[2]) == self.args.encoder_ft_start_layer
                    ):
                        status = True  # start fine-tuning from this layer

                    # update the requires_grad status is not already trainable
                    param.requires_grad = status or param.requires_grad

                # print the status of the encoder
                for (
                    name,
                    param,
                ) in self.model_without_ddp.encoder_q.named_parameters():
                    print(
                        "ecnoder_q @session {} @epoch {},".format(session, epoch),
                        name,
                        param.requires_grad,
                    )
            if epoch == self.args.pet_tuning_start_epoch:
                # Fine-tune the PET layer
                pass  # TODO handle PET tuning

        # handle trainable parameters for incremental sessions
        else:
            pass
            # Tune params as defined in config # TODO handle what to tune in the inc
            # TODO:FEAT: print all if something changes

    def update_matrix(self, accuracies: Tuple, session: int) -> None:
        """Update the accuracy matrix.

        Parameters
        ----------
        accuracies: Tuple
            Tuple of accuracies for base, incremental and all classes.
        session: int
            Current session
        """
        base_acc, inc_acc, all_acc = accuracies

        self.session_accuracies["base"][session] = max(
            base_acc,
            self.session_accuracies["base"][session],
        )
        self.session_accuracies["incremental"][session] = max(
            inc_acc,
            self.session_accuracies["incremental"][session],
        )
        self.session_accuracies["all"][session] = max(
            all_acc,
            self.session_accuracies["all"][session],
        )

    def train(self) -> None:
        """Train the model."""
        self.session_accuracies = {
            "base": [0] * self.args.sessions,
            "incremental": [0] * self.args.sessions,
            "all": [0] * self.args.sessions,
        }
        for session in range(self.args.sessions):
            # initialize dataset
            train_set, trainloader, testloader = get_dataloader(self.args, session)

            # train session
            print(f"Training session {session}...")
            print(f"Train set size: {len(train_set)}")
            print(f"Test set size: {len(testloader.dataset)}")

            if session == 0:  # base session
                if self.args.start_training_with_prototypes:
                    # replace base classifier weight with prototypes
                    print("Replacing base classifier weight with prototypes...")
                    replace_fc_with_prototypes(
                        train_set,
                        self.model_without_ddp,
                        self.args,
                        self.device_id,
                    )
                for epoch in range(self.args.epochs_base):
                    if dist_utils.is_dist_avail_and_initialized():
                        trainloader.sampler.set_epoch(epoch)
                        testloader.sampler.set_epoch(epoch)

                    # adjust learnable params
                    self.adjust_learnable_parameters(session, epoch)

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

                    if all_acc > self.session_accuracies["all"][session]:
                        self.best_model_dict = deepcopy(
                            self.model_without_ddp.state_dict(),
                        )

                    self.update_matrix((base_acc, inc_acc, all_acc), session)

                # load the best saved model for the base session
                self.model_without_ddp.load_state_dict(self.best_model_dict)

                if self.args.update_base_classifier_with_prototypes:
                    # replace base classifier weight with prototypes
                    print("Replacing base classifier weight with prototypes...")
                    replace_fc_with_prototypes(
                        train_set,
                        self.model_without_ddp,
                        self.args,
                        self.device_id,
                    )
                    base_acc, inc_acc, all_acc = test(
                        model=self.model,
                        testloader=testloader,
                        epoch=self.args.epochs_base,
                        args=self.args,
                        session=session,
                        device_id=self.device_id,
                    )

                self.update_matrix((base_acc, inc_acc, all_acc), session)

            else:
                print("Replacing inc. classifier weight with prototypes...")
                replace_fc_with_prototypes(
                    train_set,
                    self.model_without_ddp,
                    self.args,
                    self.device_id,
                )
                base_acc, inc_acc, all_acc = test(
                    model=self.model,
                    testloader=testloader,
                    epoch=0,
                    args=self.args,
                    session=session,
                    device_id=self.device_id,
                )
                self.update_matrix((base_acc, inc_acc, all_acc), session)
            print(f"Session {session} completed.")
            print("Base acc: ", self.session_accuracies["base"])
            print("Inc. acc: ", self.session_accuracies["incremental"])
            print("Overall : ", self.session_accuracies["all"])
