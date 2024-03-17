"""Test CIFAR100 dataloader."""

import os
from typing import Any

import numpy as np
import pytest
import torch

from dataloaders.helpter import get_dataloader
from tests.helper import get_mini_imagenet_dataset_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_mini_imagenet_dataset_args()),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    if os.path.exists("data/miniimagenet"):
        # Test base session data loader
        train_set, trainloader, testloader = get_dataloader(args, 0)
        train_set = train_set.dataset
        assert len(train_set) == args.base_class * args.shot
        assert len(trainloader) == np.ceil(len(train_set) / args.batch_size_base)

        data = next(iter(trainloader))
        assert data["image"][0].shape == (
            args.batch_size_base,
            3,
            args.size_crops[0],
            args.size_crops[0],
        )
        assert data["label"].shape == (args.batch_size_base,)

        # test the test data
        data = next(iter(testloader))
        assert data["image"][0].shape == (
            args.test_batch_size,
            3,
            args.size_crops[0],
            args.size_crops[0],
        )
        assert data["label"].shape == (args.test_batch_size,)

        # test incremental session
        train_set, trainloader, testloader = get_dataloader(args, 1)
        train_set = train_set.dataset
        assert len(train_set) == args.way * args.shot


test_facil_encoder(get_mini_imagenet_dataset_args())
