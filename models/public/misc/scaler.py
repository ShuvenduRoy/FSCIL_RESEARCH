"""Scaler module."""

from typing import Any, Optional

import torch
from torch import nn


class Scaler(nn.Module):
    """Scaler module."""

    def __init__(self, scale: Optional[float] = None):
        super().__init__()

        if scale is None:
            self.register_parameter("scale", nn.Parameter(torch.tensor(1.0)))
        else:
            self.scale = scale

    def forward(self, x: Any) -> Any:
        """Forward pass."""
        return x * self.scale

    def extra_repr(self) -> str:
        """Rep for print."""
        learnable = isinstance(self.scale, nn.Parameter)
        return f"scale={self.scale:.4f}, learnable={learnable}"
