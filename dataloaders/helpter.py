"""Helper functions for the methods."""

import argparse
from functools import partial
from typing import Any, Optional, Tuple

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import ViTImageProcessor

from dataloaders.datasets.cifar100 import Cifar100Dataset
from dataloaders.datasets.hf_dataset import hf_dataset
from dataloaders.sampler import DistributedEvalSampler
from utils import dist_utils


dataset_class_map = {
    "cifar100": Cifar100Dataset,
    "food101": hf_dataset,
    "caltech101": hf_dataset,
}


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
    try:
        processor = ViTImageProcessor.from_pretrained(args.hf_model_checkpoint)
        normalize = transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        )
    except Exception as e:
        print(f"Error with ViTImageProcessor: {e}")
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25],
        )

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.size_crops[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ],
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(args.size_crops[0]),
            transforms.ToTensor(),
            normalize,
        ],
    )

    return train_transforms, val_transforms


def get_dataloader(args: argparse.Namespace, session: int = 0) -> Tuple[Any, Any, Any]:
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

    trainset = dataset_class_map[args.dataset](
        root=args.dataroot,
        train=True,
        download=True,
        session=session,
        transformations=train_transforms,
        args=args,
    )
    testset = dataset_class_map[args.dataset](
        root=args.dataroot,
        train=False,
        download=False,
        session=session,
        transformations=val_transforms,
        args=args,
    )

    if args.distributed and dist_utils.is_dist_avail_and_initialized():
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            trainset,
            seed=args.seed,
            drop_last=True,
        )
        test_sampler = DistributedEvalSampler(testset, seed=args.seed)

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
        worker_init_fn=init_fn,
    )
    testloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=test_sampler,
        worker_init_fn=init_fn,
    )

    return trainset, trainloader, testloader
