"""Test CIFAR100 dataloader."""

from typing import Any

import pytest
import torch

from dataloaders.helpter import get_dataloader
from tests.helper import get_default_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (
            get_default_args()
        ),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    # Test base session data loader
    train_set, trainloader, testloader = get_dataloader(args, 0)

test_facil_encoder(get_default_args())
