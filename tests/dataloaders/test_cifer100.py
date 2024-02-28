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
    assert len(train_set) == 30000
    assert len(trainloader) == np.ceil(30000 / args.batch_size_base)
    assert len(testloader) == np.ceil(6000 / args.test_batch_size)

    data = next(iter(trainloader))
    images, labels = data
    assert images[0].shape == (
        args.batch_size_base,
        3,
        args.size_crops[0],
        args.size_crops[0],
    )
    assert labels.shape == (args.batch_size_base,)

    train_set, trainloader, testloader = get_dataloader(args, 1)
    assert len(train_set) == 25
    assert len(trainloader) == np.ceil(25 / args.batch_size_base)
    assert len(testloader) == np.ceil(500 / args.test_batch_size)


test_facil_encoder(get_default_args())
