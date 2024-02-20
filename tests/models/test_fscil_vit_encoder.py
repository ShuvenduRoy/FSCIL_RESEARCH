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
        (
            get_default_args()
        ),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    model = FSCILencoder(args)

    im_cla = torch.randn(2, 3, 224, 224)
    im_q = torch.randn(2, 3, 224, 224)
    im_k = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 100, (2,))

    embedding, logits = model.encoder_q(im_cla)
    embedding = nn.functional.normalize(embedding, dim=1)

    b = im_q.shape[0]
    embedding_q, _ = model.encoder_q(im_q)  # [b, embed_dim] [b, n_classes]
    embedding_q = nn.functional.normalize(embedding_q, dim=1)
    embedding_q = embedding_q.unsqueeze(1)  # [b, 1, embed_dim]

    # foward key
    with torch.no_grad():  # no gradient to keys
        model._momentum_update_key_encoder(True)  # update the key encoder
        embedding_k, _ = model.encoder_k(im_k)  # keys: bs x dim
        embedding_k = nn.functional.normalize(embedding_k, dim=1)

    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = (embedding_q * embedding_k.unsqueeze(1)).sum(2).view(-1, 1)

    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [embedding_q.view(-1, model.args.moco_dim), model.queue.clone().detach()])

    logits_global = torch.cat([l_pos, l_neg], dim=1)
    targets = ((labels[:, None] == model.label_queue[None, :]) & (labels[:, None] != -1)).float().to(logits_global.device)


    assert len(logits.shape) == 2
    assert logits.shape[1] == model.args.num_classes
    assert embedding.shape[1] == model.args.moco_dim