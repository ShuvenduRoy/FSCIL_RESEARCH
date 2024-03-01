"""Test the FSCIL encoder."""

import os
from typing import Any

import pytest
import torch
from torch import nn

from models.encoder import FSCILencoder
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
    for url in [
        "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
        None,
        "./checkpoint/ibot_student.pth",
        "./checkpoint/ibot_1k.pth",
        "./checkpoint/moco_v3.pth",
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    ]:
        args.pre_trained_url = url
        if url is not None and "checkpoint/" in url and not os.path.exists(url):
            continue
        model = FSCILencoder(args)

        im_cla = torch.randn(2, 3, 224, 224)
        im_q = torch.randn(2, 3, 224, 224)
        im_k = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 100, (2,))

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

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = (embedding_q * embedding_k.unsqueeze(1)).sum(2).view(-1, 1)

        # negative logits: NxK
        l_neg = torch.einsum(
            "nc,ck->nk",
            [embedding_q.view(-1, model.args.moco_dim), model.queue.clone().detach()],
        )

        logits_global = torch.cat([l_pos, l_neg], dim=1)
        targets = (
            ((labels[:, None] == model.label_queue[None, :]) & (labels[:, None] != -1))  # type: ignore
            .float()
            .to(logits_global.device)
        )

        assert targets.shape[0] == logits_global.shape[0]
        assert len(logits.shape) == 2
        assert logits.shape[1] == model.args.num_classes
        assert embedding.shape[1] == model.args.moco_dim

        # Test number of fc layer
        assert len(model.encoder_q.fc) == 2 * args.num_mlp - 1
        assert model.encoder_q.fc[0].weight.requires_grad
        assert model.encoder_q.fc[-1].weight.requires_grad

        # Test right lr for parameter
        assert model.params_with_lr[0]["lr"] == args.lr_base
        assert model.params_with_lr[1]["lr"] == args.lr_base * args.encoder_lr_factor


test_facil_encoder(get_default_args())
