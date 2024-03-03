"""Prefix module for PET."""

from typing import Any, Optional

import torch
from torch import nn

from models.public.pet.prompt import Prompt


class Prefix(nn.Module):
    """Prefix module for PET."""

    def __init__(
        self,
        length: int = 10,
        dim: int = 512,
        position: int = 1,
        key_scale: Optional[float] = None,
        val_scale: Optional[float] = None,
        compensatory: bool = True,
    ):
        super().__init__()

        self.compensatory = compensatory

        args = (length, dim, position, False)
        self.key = Prompt(*args, scale=key_scale)
        self.val = Prompt(*args, scale=val_scale)

    def forward(self, key: torch.Tensor, val: torch.Tensor) -> Any:
        """Forward function."""
        return self.key(key), self.val(val)

    def compensate(self, attn: torch.Tensor) -> torch.Tensor:
        """Compensate attention weights."""
        if not self.compensatory:
            return attn

        position, length = self.key.position, self.key.length
        s, t = position, position + length
        lamb = attn[..., s:t].sum(dim=-1, keepdim=True)
        attn1 = attn[..., :s]
        attn2 = attn[..., s:t] / lamb.clamp(min=1e-12)
        attn3 = attn[..., t:]
        return torch.cat([attn1, attn2, attn3], dim=-1)
