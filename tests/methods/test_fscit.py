"""Test the FSCIL encoder."""

from typing import Any

import pytest
import torch

from dataloaders.helpter import get_dataloader
from methods.fscit import FSCITTrainer
from methods.helper import replace_base_fc, test
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
    replace_base_fc(train_set, trainer.model, args, device_id=None)
    new_fc_weight = trainer.model.encoder_q.classifier.weight.clone()

    # new weights should not be equal to old weights
    assert not torch.equal(old_fc_weight, new_fc_weight)

    # testing the test functionality
    base_acc, inc_acc, all_acc = test(
        model=trainer.model,
        testloader=testloader,
        epoch=0,
        args=args,
        session=0,
        device_id=None,
    )
    print("Base acc: ", base_acc, "Inc acc: ", inc_acc, "total acc: ", all_acc)


test_facil_encoder(get_10way_10shot_args())
