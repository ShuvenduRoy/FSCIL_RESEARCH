"""LoRA layers for PyTorch."""

#  ---------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in
#   the repo root for license information.
#  Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#  ---------------------------------------------------------------------------------

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from models.misc.scaler import Scaler


class LinearLoRA(nn.Module):
    """Linear LoRA layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 5,
        scale: Optional[float] = None,
    ):
        super().__init__()

        assert rank > 0

        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scale = Scaler(scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A the same way as the default for nn.Linear and B to zero."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, module: nn.Linear, x: torch.Tensor) -> Any:
        """Forward function."""
        weight = self.lora_B @ self.lora_A
        return F.linear(x, module.weight + weight, module.bias)


class KVLoRA(nn.Module):
    """Linear LoRA layer with key and value."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: Union[int, Tuple[int]] = 5,
        scale: Union[None, float, Tuple[float, float]] = None,
    ):
        super().__init__()

        assert rank > 0  # type: ignore

        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros((rank, in_features))) for _ in range(2)],  # type: ignore
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros((out_features, rank))) for _ in range(2)],  # type: ignore
        )

        if not isinstance(scale, tuple):
            scale = (scale, scale)  # type: ignore
        self.scale = nn.ModuleList([Scaler(scale[0]), Scaler(scale[1])])  # type: ignore

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A the same way as the default for nn.Linear and B to zero."""
        for i in range(2):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])

    def forward(self, module: nn.Linear, x: torch.Tensor) -> Any:
        """Forward function."""
        items = zip(self.scale, self.lora_A, self.lora_B)
        weight = torch.cat([s(B @ A) for s, A, B in items], dim=0)
        zeros = torch.zeros_like(module.weight)[: -weight.shape[0]]
        weight = torch.cat([zeros, weight], dim=0)
        return F.linear(x, module.weight + weight, module.bias)


class Conv2dLoRA(nn.Module):
    """Conv2d LoRA layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 5,
        scale: Optional[float] = None,
    ):
        super().__init__()

        self.lora_A = nn.Parameter(
            torch.zeros((rank * kernel_size, in_channels * kernel_size)),
        )
        self.lora_B = nn.Parameter(
            torch.zeros((out_channels * kernel_size, rank * kernel_size)),
        )
        self.scale = Scaler(scale)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A the same way as the default for nn.Linear and B to zero."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, module: nn.Conv2d, x: torch.Tensor) -> Any:
        """Forward function."""
        weight = self.scale(self.lora_B @ self.lora_A).view(module.weight.shape)
        return F.conv2d(
            x,
            module.weight + weight,
            module.bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
