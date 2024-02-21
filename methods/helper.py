"""Traing helper module."""

import argparse
from typing import Any

import torch
from torch import nn


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
