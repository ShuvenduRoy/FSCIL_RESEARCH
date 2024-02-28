"""Traing helper module."""

import argparse
from typing import Any, Tuple

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


def get_optimizer_base(model: Any, args: argparse.Namespace) -> Tuple[Any, Any]:
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
        scheduler: torch.optim.lr_scheduler._LRScheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.step,
                gamma=args.gamma,
            )
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
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()


def test(  # noqa: PLR0915
    model: Any,
    testloader: Any,
    epoch: int,
    args: argparse.Namespace,
    session: int,
    device_id: Any,
) -> Any:
    """Test the model.

    Parameters
    ----------
    model: Any
        The model to test
    testloader: Any
        Dataloader for testing
    epoch: int
        Current epoch
    args: argparse.Namespace
        Training arguments
    session: int
        Current session
    device_id: Any
        Device id

    Returns
    -------
    Val accuracies on base, new and overall classes
    """
    model = model.eval()

    vl = Averager()
    va = Averager()

    base_labels = []
    new_labels = []
    base_preds = []
    new_preds = []
    class_acc = {}  # Dictionary to store class-wise accuracy
    class_count = {}  # Dictionary to store count of samples per class

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
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
            logits = model(data[0])

            # calculate the loss
            loss = F.cross_entropy(logits, labels)

            # print the loss and accuracy
            acc = count_acc(logits.detach(), labels)

            vl.add(loss.item())
            va.add(acc)

            # select samples with labels less than args.base_class
            try:
                test_label_base = labels[labels < args.base_class]
                agg_preds_base = logits[labels < args.base_class]
                base_labels.append(test_label_base.detach().cpu())
                base_preds.append(agg_preds_base.detach().cpu())

                test_label_new = labels[labels >= args.base_class]
                agg_preds_new = logits[labels >= args.base_class]
                new_labels.append(test_label_new.detach().cpu())
                new_preds.append(agg_preds_new.detach().cpu())
            except Exception:
                pass

    base_labels = torch.cat(base_labels, dim=0)
    new_labels = torch.cat(new_labels, dim=0)
    base_preds = torch.cat(base_preds, dim=0)
    new_preds = torch.cat(new_preds, dim=0)
    all_labels = torch.cat([base_labels, new_labels], dim=0)
    all_preds = torch.cat([base_preds, new_preds], dim=0)

    base_acc = count_acc(base_preds, base_labels)
    new_acc = count_acc(new_preds, new_labels)
    print(
        "epo {}, test, loss={:.4f} acc={:.4f} base acc={:.4f} new acc={:.4f},".format(
            epoch,
            vl.item(),
            va.item(),
            base_acc,
            new_acc,
        ),
    )

    # Calculate per-class accuracy
    for class_index in range(args.base_class + session * args.way):
        class_mask = all_labels == class_index
        class_preds = all_preds[class_mask]
        class_labels = all_labels[class_mask]
        class_acc[class_index] = count_acc(class_preds, class_labels)
        class_count[class_index] = len(class_labels)

    # Print per-class accuracy
    for class_index in range(args.base_class + session * args.way):
        class_type = "Base" if class_index < args.base_class else "New"
        class_accuracy = class_acc[class_index]
        class_samples = class_count[class_index]
        print(
            f"{class_type} Class {class_index}: Accuracy={class_accuracy:.4f}, Samples={class_samples}",
        )

    # clculate per-session accuracy
    session_acc = {}
    for session_index in range(session):
        session_start_class = args.base_class + (session_index + 1) * args.way
        session_end_class = session_start_class + args.way
        session_labels = all_labels[
            (all_labels >= session_start_class) & (all_labels < session_end_class)
        ]
        session_preds = all_preds[
            (all_labels >= session_start_class) & (all_labels < session_end_class)
        ]
        session_acc[session_index] = count_acc(session_preds, session_labels)
        print(f"Session {session_index}: Accuracy={session_acc[session_index]:.4f}")
    return base_acc, new_acc, va.item()  # TODO add all reduce for distributed training


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
