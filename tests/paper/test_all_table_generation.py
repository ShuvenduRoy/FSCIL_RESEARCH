"""Test the FSCIL encoder."""

import os
from typing import Any

import pytest
import torch

from paper.table_generators.ablation_table_generator import (
    ablation_table_generator,
)
from paper.table_generators.appendix_naive_baseline_all_encoders import (
    generate_naive_baseline_all_encoders,
)
from paper.table_generators.main_and_appendix_baseline_all import (
    main_generate_all_baselines,
)
from paper.table_generators.main_fscit import main_fscit
from paper.table_generators.main_naive_baseline_all_encoders import (
    main_generate_naive_baseline_all_encoders,
)


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
        main_generate_naive_baseline_all_encoders()
        generate_naive_baseline_all_encoders()
        main_generate_all_baselines()
        ablation_table_generator()
        main_fscit()


test_facil_encoder(None)
