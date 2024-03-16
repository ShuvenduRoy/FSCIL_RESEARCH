"""Test food101 dataloader."""

from typing import Any

import numpy as np
import pytest
import torch

from dataloaders.helpter import get_dataloader
from tests.helper import get_caltech101_dataset_args, get_food101_dataset_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_caltech101_dataset_args()),
        (get_food101_dataset_args()),
    ],
)
def test_hf_dataset(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    # Test base session data loader
    train_set, trainloader, testloader = get_dataloader(args, 0)
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

    data = next(iter(testloader))
    assert data["image"][0].shape == (
        args.test_batch_size,
        3,
        args.size_crops[0],
        args.size_crops[0],
    )
    assert data["label"].shape == (args.test_batch_size,)

    train_set, trainloader, testloader = get_dataloader(args, 1)
    assert len(train_set) == args.way * args.shot


test_hf_dataset(get_caltech101_dataset_args())
