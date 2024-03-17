"""Test CIFAR100 dataloader."""

from typing import Any

import numpy as np
import pytest
import torch

from dataloaders.helpter import get_dataloader
from tests.helper import get_default_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_default_args()),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    # Test base session data loader
    train_set, trainloader, testloader = get_dataloader(args, 0)
    train_set = train_set.dataset
    assert len(train_set) == args.base_class * args.shot
    assert len(trainloader) == np.ceil(
        args.base_class * args.shot / args.batch_size_base,
    )
    assert len(testloader) == np.ceil(6000 / args.test_batch_size)

    data = next(iter(trainloader))
    images, labels = data["image"], data["label"]
    assert images[0].shape == (
        args.batch_size_base,
        3,
        args.size_crops[0],
        args.size_crops[0],
    )
    assert labels.shape == (args.batch_size_base,)

    train_set, trainloader, testloader = get_dataloader(args, 1)
    train_set = train_set.dataset
    assert len(train_set) == 25
    assert len(trainloader) == np.ceil(25 / args.batch_size_base)
    assert len(testloader) == np.ceil(
        100 * (args.base_class + args.shot) / args.test_batch_size,
    )


test_facil_encoder(get_default_args())
