"""Test CIFAR100 dataloader."""

from typing import Any

import numpy as np
import pytest
import torch

from dataloaders.helpter import get_dataloader
from tests.helper import get_cifar_fscit_args, get_default_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_default_args()),
        (get_cifar_fscit_args()),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    # Test base session data loader
    train_set, trainloader, testloader = get_dataloader(args, 0)
    train_set = train_set.dataset
    if args.fsl_setup == "FSCIL":
        assert len(train_set) == 30000
        assert len(testloader) == np.ceil(6000 / args.test_batch_size)
    else:
        assert len(train_set) == args.base_class * args.shot
        assert len(trainloader) == np.ceil(
            args.base_class * args.shot / args.batch_size_base,
        )
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

    if args.fsl_setup == "FSCIT":
        assert len(trainloader) == np.ceil(50 / args.batch_size_base)
        assert len(testloader.dataset) == 2000
    else:
        assert len(train_set) == 2500
        assert len(testloader.dataset) == 6500


test_facil_encoder(get_cifar_fscit_args())
test_facil_encoder(get_default_args())
