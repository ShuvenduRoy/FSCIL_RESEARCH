"""Supervised Contrastive Loss Modified from UniMoCo."""

import torch
from torch import nn


class SupContrastive(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, reduction: str = "mean"):
        """Init.

        Parameters
        ----------
        reduction: str
            Reduction term.
        """
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Parameters
        ----------
        y_pred: torch.tensor
            prediction embedding vectors
        y_true: torch.tensor
            actual class labels

        Returns
        -------
        loss (torch.tensor)
        """
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = y_true * torch.exp(torch.neg(y_pred))
        num_pos = torch.sum(y_true, dim=1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == "mean":
            return torch.mean(loss)
        return loss
