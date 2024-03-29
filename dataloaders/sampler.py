"""Sampler for distributed evaluation.

Acknowledgement: The original author of this script is Seungjun Nah
See: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
and https://discuss.pytorch.org/t/how-to-validate-in-distributeddataparallel-correctly/94267/11

"""

from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler):
    """Sampler for distributed evaluation.

    DistributedEvalSampler is different from DistributedSampler. It does NOT add
    extra samples to make it evenly divisible.

    DistributedEvalSampler should NOT be used for training. The distributed processes
    could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584

    Shuffle is disabled by default.

    DistributedEvalSampler is for evaluation purpose where synchronization does
    not happen every epoch. Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset. It is especially
    useful in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`.

    Dataset is assumed to be of constant size.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset used for sampling.
    num_replicas : int, optional, default=`None`
        Number of processes participating in distributed training. By
        default, :attr:`rank` is retrieved from the current distributed group.
    rank : int, optional, default=`None`
        Rank of the current process within :attr:`num_replicas`. By default,
        :attr:`rank` is retrieved from the current distributed group.
    shuffle (bool, optional):
        If ``True`` (default), sampler will shuffle the indices.
    seed : int, optional, default=0
        Random seed used to shuffle the sampler if :attr:`shuffle=True`.
        This number should be identical across all processes in the
        distributed group.

    Warnings
    --------
    In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>`
    method at the beginning of each epoch **before** creating the
    :class:`DataLoader` iterator is necessary to make shuffling work properly
    across multiple epochs. Otherwise, the same ordering will be always used.


    Examples
    --------
    >>> sampler = DistributedSampler(dataset) if is_distributed else None
    >>> loader = DataLoader(dataset, shuffle=(sampler is None),
    ...                     sampler=sampler)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     if is_distributed:
    ...         sampler.set_epoch(epoch)
    ...     train(loader)

    """

    def __init__(
        self,
        dataset: Any,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """Initialize the sampler."""
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = len(self.dataset)  # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)  # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator that iterates over the indices of the dataset."""
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random
        ordering for each epoch. Otherwise, the next iteration of this sampler
        will yield the same ordering.

        Parameters
        ----------
        epoch : int
            Epoch number.

        """
        self.epoch = epoch
