"""Test the FSCIL encoder."""

from typing import Any

import pytest
import torch


torch.manual_seed(42)


@pytest.mark.parametrize(
    "args",
    [
        (None),
    ],
)
def test_facil_encoder(args: Any) -> None:
    """Test ExponentialMovingAverage."""
    from paper.table_generators.appendix_naive_baseline_all_encoders import (
        generate_naive_baseline_all_encoders,
    )

    generate_naive_baseline_all_encoders()
