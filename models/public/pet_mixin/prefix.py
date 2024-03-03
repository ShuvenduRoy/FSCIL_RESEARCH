"""PrefixMixin class."""

from typing import Any, Optional, Tuple

from models.public.pet.prefix import Prefix


class PrefixMixin:
    """PrefixMixin class."""

    prefix: Prefix

    def attach_prefix(self, prefix: Prefix) -> None:
        """Attach prefix to the model."""
        self.prefix = prefix

    def detach_prefix(self) -> Optional[Prefix]:
        """Detach prefix from the model."""
        prefix, self.prefix = getattr(self, "prefix", None), None  # type: ignore
        return prefix

    def add_prefix(self, key: Any, val: Any) -> Tuple[Any, Any]:
        """Add prefix to the key and value."""
        if isinstance(getattr(self, "prefix", None), Prefix):
            key, val = self.prefix(key, val)
        return key, val

    def compensate_prefix(self, attn: Any) -> Any:
        """Compensate prefix from the attention."""
        if isinstance(getattr(self, "prefix", None), Prefix):
            attn = self.prefix.compensate(attn)
        return attn
