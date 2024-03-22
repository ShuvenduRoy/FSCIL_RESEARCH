"""Test the FSCIL encoder."""

import os
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
    if os.path.exists("results"):
        from paper.table_generators.main_naive_baseline_all_encoders import (
            main_generate_naive_baseline_all_encoders,
        )

        main_generate_naive_baseline_all_encoders()

        from paper.table_generators.appendix_naive_baseline_all_encoders import (
            generate_naive_baseline_all_encoders,
        )

        generate_naive_baseline_all_encoders()

        from paper.table_generators.main_and_appendix_baseline_all import (
            main_generate_all_baselines,
        )

        main_generate_all_baselines()


test_facil_encoder(None)