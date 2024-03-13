"""Tests for the SupContrastive class."""

import unittest

import torch

from losses.contrastive import SupConLoss


# Set seed for deterministic output
torch.manual_seed(42)


class TestSupContrastive(unittest.TestCase):
    """Test the SupContrastive class."""

    def test_forward(self) -> None:
        """Test the forward method."""
        loss = SupConLoss()

        # Create deterministic inputs
        y_pred_1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        y_pred_2 = torch.tensor([[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]])
        y_pred = torch.stack([y_pred_1.unsqueeze(1), y_pred_2.unsqueeze(1)], dim=1)
        y_true = torch.tensor([0, 1])

        # Call the forward method
        result = loss.forward(y_pred, y_true)
        print(result)

        # Check the result
        expected_result = torch.tensor([6.4681])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
