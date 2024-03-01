"""Tests for the SupContrastive class."""

import unittest

import torch

from losses.contrastive import SupContrastive


# Set seed for deterministic output
torch.manual_seed(42)


class TestSupContrastive(unittest.TestCase):
    """Test the SupContrastive class."""

    def test_forward(self) -> None:
        """Test the forward method."""
        model = SupContrastive(reduction="mean")

        # Create deterministic inputs
        y_pred = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        y_true = torch.tensor([[1, 0, 0], [0, 1, 0]])

        # Call the forward method
        result = model.forward(y_pred, y_true)

        # Check the result
        expected_result = torch.tensor([1.1519])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
