"""Test the FSCIL encoder."""

from typing import Any

import pytest
import torch

from dataloaders.helpter import get_dataloader
from methods.fscit import FSCITTrainer
from methods.helper import replace_base_fc
from tests.helper import get_10way_10shot_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_10way_10shot_args()),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    trainer = FSCITTrainer(args)
    train_set, trainloader, testloader = get_dataloader(args, 0)

    # testing replace_base_fc
    old_fc_weight = trainer.model.encoder_q.classifier.weight.clone()
    replace_base_fc(train_set, trainer.model_without_ddp, args, device_id=None)
    new_fc_weight = trainer.model.encoder_q.classifier.weight.clone()

    # new weights should not be equal to old weights
    assert not torch.equal(old_fc_weight, new_fc_weight)


test_facil_encoder(get_10way_10shot_args())
