"""Traing helper module."""

import argparse
from typing import Any

import torch
from torch import nn


class Averager:
    """Average meter."""

    def __init__(self) -> None:
        """Init function."""
        self.n = 0
        self.v = 0

    def add(self, x: Any) -> None:
        """Add value.

        Parameters
        ----------
        x: Any
            Value to add
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self) -> float:
        """Return the average.

        Returns
        -------
        average
        """
        return self.v


def get_optimizer_base(model: nn.Module, args: argparse.Namespace) -> tuple[Any, Any]:
    """Return the optimizer for FSCIL training.

    Parameters
    ----------
    mdoel: nn.Module
        The trainable model
    args: argparse.Namespace
        arguments

    Returns
    -------
    Tuple[optimizer, scheduler]
    """
    optimizer = torch.optim.SGD(
        model.params_with_lr,
        args.lr_base,
        momentum=0.9,
        nesterov=True,
        weight_decay=args.decay,
    )
    if args.schedule == "Step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step,
            gamma=args.gamma,
        )
    elif args.schedule == "Milestone":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.gamma,
        )
    elif args.schedule == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_base,
        )

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    trainloader: Any,
    criterion: nn.Module,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    args: argparse.Namespace,
) -> Any:
    """One epoch of training of the model.

    Parameters
    ----------
    model: nn.Module
        The model to train
    trainloader: Any
        Dataloader for training
    criterion: nn.Module
        Loss function
    optimizer: Any
        Model optimizer
    scheduler: Any
        LR scheduler
    epoch: int
        Current training epoch
    """
    pass
