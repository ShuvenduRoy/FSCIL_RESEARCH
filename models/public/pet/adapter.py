"""Adapter Module."""

import math
from typing import Any, Optional, Type, Union

from torch import nn

from models.public.misc.scaler import Scaler


class Adapter(nn.Module):
    """Adapter Module."""

    def __init__(
        self,
        embed_dim: int,
        down_sample: Union[float, int] = 5,
        mode: str = "parallel",  # enum before, after, parallel
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = embed_dim * down_sample
        hidden_dim = int(hidden_dim)

        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, embed_dim),
            Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module: Any, x: Any, **kwargs: Any) -> Any:
        """Forward function."""
        if self.mode == "before":
            return module(self.layer(x) + x, **kwargs)
        if self.mode == "after":
            return self.layer(module(x, **kwargs)) + x
        return module(x, **kwargs) + self.layer(x)


class Conv2dAdapter(nn.Module):
    """Conv2d Adapter Module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        down_sample: Union[float, int] = 5,
        mode: str = "before",  # enum before, after, parallel
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = in_channels * down_sample
        hidden_dim = int(hidden_dim)

        if out_channels is None:
            out_channels = in_channels

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            act_layer(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            Scaler(scale),
        )
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module: Any, x: Any, **kwargs: Any) -> Any:
        """Forward function."""
        if self.mode == "before":
            return module(self.layer(x) + x, **kwargs)
        if self.mode == "after":
            return self.layer(module(x, **kwargs)) + x
        return module(x, **kwargs) + self.layer(x)
