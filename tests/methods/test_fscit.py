"""Test the FSCIL encoder."""

from typing import Any

import pytest
import torch

from dataloaders.helpter import get_dataloader
from methods.fscit import FSCITTrainer
from methods.helper import replace_base_fc
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
    trainer = FSCITTrainer(args)
    train_set, _, _ = get_dataloader(args, 1)
    old_fc_weight = trainer.model.encoder_q.classifier.weight.clone()

    # testing replace_base_fc
    replace_base_fc(train_set, trainer.model, args, device_id=None)
    new_fc_weight = trainer.model.encoder_q.classifier.weight.clone()

    # new weights should not be equal to old weights
    assert not torch.equal(old_fc_weight, new_fc_weight)


test_facil_encoder(get_default_args())
