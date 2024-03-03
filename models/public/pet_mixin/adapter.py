"""AdapterMixin Module."""

from typing import Any, Dict, List

import torch
from torch import nn

from models.public.pet import Adapter, Conv2dAdapter


NAME_SEP = "/"


def normalize_name(name: str) -> str:
    """Normalize name by replacing . with NAME_SEP."""
    return name.replace(".", NAME_SEP)


def denormlize_name(name: str) -> str:
    """Denormalize name by replacing NAME_SEP with ."""
    return name.replace(NAME_SEP, ".")


def get_submodule(
    module: nn.Module,
    name: str,
    default: Any = nn.Identity(),  # noqa: B008
) -> nn.Module:
    """Get submodule from the module."""
    names = name.split(NAME_SEP)
    while names:
        module = getattr(module, names.pop(0), default)
    return module


class AdapterMixin:
    """AdapterMixin class."""

    adapters: nn.ModuleDict

    def attach_adapter(self, **kwargs: Dict[str, nn.Module]) -> None:
        """Attach adapter to the model."""
        if not isinstance(getattr(self, "adapters", None), nn.ModuleDict):
            self.adapters = nn.ModuleDict()
        for name, adapter in kwargs.items():
            name = normalize_name(name)  # noqa: PLW2901
            self.adapters.add_module(name, adapter)  # type: ignore

    def detach_adapter(self, *names: List[str]) -> Any:
        """Detach adapter."""
        adapters: dict = {}
        if not hasattr(self, "adapters"):
            return adapters

        names = names if names else map(denormlize_name, self.adapters.keys())  # type: ignore
        for name in names:
            adapters[name] = self.adapters.pop(normalize_name(name))  # type: ignore
        return adapters

    def adapt_module(self, name: str, x: torch.Tensor, **kwargs: Any) -> Any:
        """Adapt module."""
        name = normalize_name(name)
        module = get_submodule(self, name)  # type: ignore
        if not isinstance(module, (Adapter, Conv2dAdapter)):
            assert kwargs == {}, f"Unknown kwargs: {kwargs.keys()}"

        if hasattr(self, "adapters") and name in self.adapters:
            return self.adapters[name](module, x, **kwargs)
        return module(x, **kwargs)
