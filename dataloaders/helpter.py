"""Helper functions for the methods."""
import argparse
from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dataloaders.datasets.cifar100 import Cifar100Dataset
from dataloaders.sampler import DistributedEvalSampler
from utils import dist_utils


def get_transform(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Return the transforms for the dataset.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Arguments passed to the trainer.

    Returns
    -------
    Tuple[Any, Any]
        The crop transform and the secondary transform for the dataset.
    """
    if args.dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
    if args.dataset == "cub200":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.dataset == "mini_imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.size_crops[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])
    val_transforms = transforms.Compose([
                transforms.Resize(args.size_crops[0]),
                transforms.ToTensor(),
                normalize,
            ])

    return train_transforms, val_transforms

def get_dataloader(args: argparse.Namespace,
                   session: int = 0) -> Tuple[Any, Any, Any]:
    """Get the base dataloader.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Arguments passed to the trainer.
    session : int, optional
        The current session.

    Returns
    -------
    Tuple[Any, Any, Any]
        The trainset, trainloader, and testloader for the base classes.
    """
    train_transforms, val_transforms = get_transform(args)

    if args.dataset == "cifar100":
        trainset = Cifar100Dataset(root=args.dataroot,
                                    train=True,
                                    download=True,
                                    session=session,
                                    transformations=train_transforms,
                                    args=args)
        testset = Cifar100Dataset(root=args.dataroot,
                                  train=False,
                                  download=False,
                                  session=session,
                                  transformations=val_transforms,
                                  args=args)

    if args.dataset == "cub200":
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                       crop_transform=crop_transform, secondary_transform=secondary_transform,
                                       rotation_pred=args.rotation_pred, args=args)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, args=args)

    if args.dataset == "mini_imagenet":
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                             crop_transform=crop_transform, secondary_transform=secondary_transform,
                                             rotation_pred=args.rotation_pred, args=args)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, args=args)


    if args.distributed and dist_utils.is_dist_avail_and_initialized():
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            trainset,
            seed=args.seed,
            drop_last=True,
        )
        test_sampler = DistributedEvalSampler(testset,
                                              seed=args.seed)

        init_fn = (
            partial(
                dist_utils.worker_init_fn,
                num_workers=args.num_workers,
                rank=dist_utils.get_rank(),
                seed=args.seed,
            )
            if args.seed is not None
            else None
        )
    else:
        init_fn, train_sampler, test_sampler = None, None, None

    trainloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size_base,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        worker_init_fn=init_fn)
    testloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=test_sampler,
        worker_init_fn=init_fn)

    return trainset, trainloader, testloader


def get_session_classes(args: argparse.Namespace, session: int) -> np.ndarray:
    """Get the classes for the current session.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the trainer.
    session : int
        The current session.

    Returns
    -------
    np.ndarray
        The classes for the current session.
    """
    return np.arange(args.base_class + session * args.way)
