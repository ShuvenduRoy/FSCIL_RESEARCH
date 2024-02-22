"""Traing helper module."""

import argparse
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa
from tqdm import tqdm


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


def get_optimizer_base(model: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """Return the optimizer for FSCIL training.

    Parameters
    ----------
    mdoel: Any (nn.Module)
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


def count_acc(logits: torch.tensor, label: torch.tensor) -> float:
    """Count the accuracy of the model.

    Parameters
    ----------
    logits: torch.tensor
        The model logits
    label: torch.tensor
        The actual labels

    Returns
    -------
    accuracy
    """
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).mean().item()
    return (pred == label).mean().item()


def train_one_epoch(
    model: Any,
    trainloader: Any,
    criterion: nn.Module,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    args: argparse.Namespace,
    device_id: Any,
) -> None:
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
    args: argparse.Namespace
        Training arguments
    device_id: Any
        Device id

    Returns
    -------
    None
    """
    tl = Averager()
    tl_ce = Averager()
    tl_moco = Averager()
    ta = Averager()

    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for _, batch in enumerate(tqdm_gen, 1):
        data, labels = batch
        labels = labels.long()
        if device_id is not None:
            for i in range(len(data)):
                data[i] = data[i].cuda(device_id, non_blocking=True)
            labels = labels.cuda(device_id, non_blocking=True)
        elif torch.cuda.is_available():
            for i in range(len(data)):
                data[i] = data[i].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # model foward pass
        logits, _, logits_global, targets_global = model(data[0], data[1], labels)

        # calculate the loss
        moco_loss = criterion(logits_global, targets_global)
        ce_loss = F.cross_entropy(logits, labels)
        loss = args.ce_loss_factor * ce_loss + args.moco_loss_factor * moco_loss
        if torch.isnan(loss):
            raise Exception("Loss is NaN")

        # update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print the loss and accuracy
        acc = count_acc(logits.detach(), labels)
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            "Session 0, epo {}, lrc={:.4f}, total loss={:.4f} moco loss={:.4f} ce loss={:.4f} acc={:.4f}".format(
                epoch,
                lrc,
                loss.item(),
                moco_loss.item(),
                ce_loss.item(),
                acc,
            ),
        )
        tl.add(loss.item())
        tl_moco.add(moco_loss.item())
        tl_ce.add(ce_loss.item())
        ta.add(acc)
