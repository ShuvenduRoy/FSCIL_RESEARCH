"""Prompt module for PET."""

from typing import Optional

import torch
from torch import nn

from models.public.misc.scaler import Scaler


class Prompt(nn.Module):
    """Prompt module for PET."""

    def __init__(
        self,
        length: int = 5,
        dim: int = 512,
        position: int = 1,
        reducible: bool = False,
        scale: Optional[float] = 1.0,
    ):
        super().__init__()

        self.length = length
        self.dim = dim
        self.position = position
        self.reducible = reducible

        tokens = nn.Parameter(torch.zeros(length, dim))
        self.register_parameter("tokens", tokens)
        nn.init.uniform_(self.tokens.data, 0, 0.01)  # type: ignore

        self.scaler = Scaler(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        assert x.shape[-1] == self.dim

        tokens = self.scaler(self.tokens).expand(x.shape[0], -1, -1)
        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position :]
            return torch.cat([x1, tokens, x2], dim=1)
        return torch.cat([tokens, x], dim=1)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the prompt tokens."""
        if not self.reducible:
            return x

        if self.position > 0:
            x1, x2 = x[:, : self.position], x[:, self.position + self.length :]
            return torch.cat([x1, x2], dim=1)
        return x[:, self.length :]

    def extra_repr(self) -> str:
        """Repr."""
        tpl = "length={}, dim={}, position={}, reducible={}"
        return tpl.format(self.length, self.dim, self.position, self.reducible)
