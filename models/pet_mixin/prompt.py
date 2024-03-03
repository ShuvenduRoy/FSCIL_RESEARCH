"""PromptMixin module."""

from typing import Any

from models.pet.prompt import Prompt


class PromptMixin:
    """PromptMixin class."""

    prompt: Prompt

    def attach_prompt(self, prompt: Prompt) -> None:
        """Attach prompt to the model."""
        self.prompt = prompt

    def detach_prompt(self) -> Any:
        """Detach prompt from the model."""
        prompt, self.prompt = getattr(self, "prompt", None), None  # type: ignore
        return prompt

    def add_prompt(self, x: Any) -> Any:
        """Add prompt to the input."""
        if isinstance(getattr(self, "prompt", None), Prompt):
            x = self.prompt(x)
        return x

    def reduce_prompt(self, x: Any) -> Any:
        """Reduce prompt from the input."""
        if isinstance(getattr(self, "prompt", None), Prompt):
            x = self.prompt.reduce(x)
        return x
