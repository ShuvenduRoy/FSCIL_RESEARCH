"""Contains a collection of utility functions for the project."""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def str2bool(v: str) -> bool:
    """Convert a string to a boolean value.

    Parameters
    ----------
    v : str
        The string to convert.

    Returns
    -------
    bool: The boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def calculate_cdf(prototypes: List[np.ndarray]) -> None:
    """Create a list to store inter-class distances.

    Parameters
    ----------
    prototypes : list
        A list of class prototypes,
        where each prototype is a 1D numpy array.

    Returns
    -------
    None
    """
    inter_class_distances = []

    # Calculate inter-class distances for all pairs of prototypes
    for i in range(len(prototypes)):
        for j in range(i + 1, len(prototypes)):
            cosine_sim = cosine_similarity([prototypes[i]], [prototypes[j]])[0][0]
            inter_class_distance = 1 - cosine_sim
            inter_class_distances.append(inter_class_distance)

    # Sort the distances
    sorted_distances = np.sort(inter_class_distances)

    # Calculate the cumulative distances
    cumulative_distances = np.arange(1, len(sorted_distances) + 1) / len(
        sorted_distances,
    )

    # Plot the CDF
    plt.plot(sorted_distances, cumulative_distances)
    plt.xlabel("Inter-class Distance")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distance Function (CDF) for 100 Class Prototypes")
    plt.grid()
    plt.show()


def pseudo_intra_class_cdf() -> None:
    """Calculate intra-class distances and plot cumulative distance function (CDF)."""
    # Sample embeddings and their corresponding labels
    embeddings = np.random.rand(
        100,
        4,
    )  # Assuming 100 samples with 4-dimensional embeddings
    labels = np.random.randint(0, 10, size=100)  # Assuming 10 classes

    # Prototypes for each class (assuming 10 classes and 4-dimensional prototypes)
    prototypes = [np.random.rand(4) for _ in range(10)]

    # Create a dictionary to store intra-class distances for each class
    intra_class_distances: Dict[int, List[float]] = {
        class_label: [] for class_label in set(labels)
    }

    # Calculate intra-class distances for each sample
    for i in range(len(embeddings)):
        sample = embeddings[i]
        class_label = labels[i]
        prototype = prototypes[class_label]

        # Calculate cosine similarity
        cosine_sim = cosine_similarity([sample], [prototype])[0][0]

        # Calculate intra-class distance
        intra_class_distance = 1 - (cosine_sim / len(prototypes))

        # Append to the corresponding class's list
        intra_class_distances[class_label].append(intra_class_distance)

    for key, value in intra_class_distances.items():
        # average
        intra_class_distances[key] = np.mean(value)

    distances = intra_class_distances.values()
    sorted_distances = np.sort(list(distances))
    cumulative_distances = np.arange(1, len(prototypes) + 1) / len(prototypes)
    plt.plot(sorted_distances, cumulative_distances, label="Overall")

    plt.xlabel("Intra-class Distance")
    plt.ylabel("Cumulative Probability")
    plt.title("Intra-class Cumulative Distance Function (CDF)")
    plt.legend()
    plt.grid()
    plt.show()


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed.

    Returns
    -------
        None
    """
    if seed == 0:
        print(" random seed")
        torch.backends.cudnn.benchmark = True
    else:
        print("manual seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args: argparse.Namespace) -> int:
    """Set the GPU to use.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    int: The number of GPUs to use.
    """
    gpu_list = [int(x) for x in args.gpu.split(",")]
    print("use gpu:", gpu_list)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return gpu_list.__len__()


def ensure_path(path: str) -> None:
    """Ensure the path exists.

    Parameters
    ----------
    path : str
        The path to ensure.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)


class Averager:
    """A to calculate the average of a series of numbers."""

    def __init__(self) -> None:
        """Initialize the Averager class."""
        self.n = 0.0  # Change the type of self.n from int to float
        self.v = 0.0

    def add(self, x: float) -> None:
        """Add a number to the series.

        Parameters
        ----------
        x : float
            The number to add.

        Returns
        -------
        None
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self) -> float:
        """Return the average of the series.

        Returns
        -------
        float: The average of the series.
        """
        return self.v


def count_acc(logits: torch.Tensor, label: torch.Tensor) -> float:
    """Calculate the accuracy of the model.

    Parameters
    ----------
    logits : torch.Tensor
        The model's predictions.
    label : torch.Tensor
        The true labels.

    Returns
    -------
    float: The accuracy of the model.
    """
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()
