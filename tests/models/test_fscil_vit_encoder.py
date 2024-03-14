"""Test the FSCIL encoder."""

from typing import Any

import pytest
import torch
from torch import nn

from models.encoder import FSCILencoder
from tests.helper import get_default_args, get_lora_args


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (get_default_args()),
        (get_lora_args()),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    model = FSCILencoder(args)

    im_cla = torch.randn(2, 3, 224, 224)
    im_q = torch.randn(2, 3, 224, 224)
    im_k = torch.randn(2, 3, 224, 224)

    patch_embed, embedding, logits = model.encoder_q(im_cla)
    assert len(patch_embed.shape) == 2
    embedding = nn.functional.normalize(embedding, dim=1)

    _, embedding_q, _ = model.encoder_q(im_q)  # [b, embed_dim] [b, n_classes]
    embedding_q = nn.functional.normalize(embedding_q, dim=1)
    embedding_q = embedding_q.unsqueeze(1)  # [b, 1, embed_dim]

    # foward key
    with torch.no_grad():  # no gradient to keys
        model._momentum_update_key_encoder(True)  # update the key encoder
        _, embedding_k, _ = model.encoder_k(im_k)  # keys: bs x dim
        embedding_k = nn.functional.normalize(embedding_k, dim=1)

    assert len(logits.shape) == 2
    assert logits.shape[1] == model.args.num_classes
    assert embedding.shape[1] == model.args.moco_dim

    # Test number of fc layer
    assert len(model.encoder_q.mlp) == 2 * args.num_mlp - 1
    assert model.encoder_q.mlp[0].weight.requires_grad
    assert model.encoder_q.mlp[-1].weight.requires_grad

    # Test right lr for parameter
    assert model.params_with_lr[0]["lr"] == args.lr_base
    assert model.params_with_lr[1]["lr"] == args.lr_base * args.encoder_lr_factor


test_facil_encoder(get_default_args())
