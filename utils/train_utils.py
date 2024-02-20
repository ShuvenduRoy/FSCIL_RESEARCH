"""Contains a collection of utility functions for the project."""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def override_training_configs(args: argparse.Namespace) -> argparse.Namespace:
    """Override the training configurations.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    argparse.Namespace: The command-line arguments
    with the overridden training configurations.
    """
    args.exp_name = args.dataset + "_" + args.exp_name
    if args.pre_trained_url == "None":
        args.pre_trained_url = None
    if args.adapt_blocks < 0:
        args.adapt_blocks *= -1
        args.adapt_blocks = list(range(args.adapt_blocks, 12))
    else:
        args.adapt_blocks = list(range(args.adapt_blocks))

    args.save_path = os.path.join("checkpoint", args.exp_name)

    if args.encoder == "vit-16":
        args.size_crops = [224, 224]
    if os.path.exists("/home/sneha/"):
        args.num_workers = 4
    return args


def get_dataset_configs(args: argparse.Namespace) -> argparse.Namespace:
    """Set up the datasets.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.

    Returns
    -------
    argparse.Namespace: The command-line arguments
    with the datasets configs.
    """
    if args.dataset == "cifar100":
        from datasets.cifar100 import Cifar100Dataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.Dataset = Cifar100Dataset

    elif args.dataset == "cub200":
        from datasets.cub200 import Cub200Dataset

        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
        args.size_crops = [224, 96]
        args.min_scale_crops = [0.2, 0.05]
        args.max_scale_crops = [1, 0.14]
        args.milestones = [60, 80, 100]

        args.Dataset = Cub200Dataset

    elif args.dataset == "mini_imagenet":
        from datasets.miniimagenet import MiniImagenetDataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.size_crops = [84, 50]
        args.min_scale_crops = [0.2, 0.05]
        args.max_scale_crops = [1, 0.14]
        args.milestones = [40, 70, 100]

        args.Dataset = MiniImagenetDataset
    return args


def get_command_line_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    """Get the command line parser for the script.

    Returns
    -------
        argparse.ArgumentParser: The command line parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cub200",
        choices=["mini_imagenet", "cub200", "cifar100"],
    )
    parser.add_argument("--dataroot", type=str, default="./data")

    # about pre-training
    parser.add_argument("--epochs_base", type=int, default=100)
    parser.add_argument("--epochs_new", type=int, default=10)
    parser.add_argument("--lr_base", type=float, default=0.1)
    parser.add_argument("--lr_new", type=float, default=0.1)
    parser.add_argument("--lrw", type=float, default=0.1)
    parser.add_argument("--lrb", type=float, default=0.1)
    parser.add_argument(
        "--schedule",
        type=str,
        default="Step",
        choices=["Step", "Milestone", "Cosine"],
    )
    parser.add_argument("--milestones", nargs="+", type=int, default=[60, 70])
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--temperature", type=int, default=16)
    parser.add_argument("--not_data_init", type=str2bool, default=False)
    parser.add_argument("--batch_size_base", type=int, default=64)
    parser.add_argument(
        "--batch_size_new",
        type=int,
        default=0,
        help="set 0 will use all the available training image for new",
    )
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument(
        "--base_mode",
        type=str,
        default="ft_cos",
    )
    parser.add_argument(
        "--new_mode",
        type=str,
        default="avg_cos",
    )

    # for SAVC
    parser.add_argument(
        "--moco_dim",
        default=128,
        type=int,
        help="feature dimension (default: 128)",
    )
    parser.add_argument(
        "--moco_k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )
    parser.add_argument(
        "--moco_m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco_t",
        default=0.07,
        type=float,
        help="softmax temperature (default: 0.07)",
    )
    parser.add_argument("--mlp", type=str2bool, default=False, help="use mlp head")
    parser.add_argument(
        "--num_crops",
        type=int,
        default=[2, 1],
        nargs="+",
        help="amount of crops",
    )
    parser.add_argument(
        "--size_crops",
        type=int,
        default=[32, 18],
        nargs="+",
        help="resolution of inputs",
    )
    parser.add_argument(
        "--min_scale_crops",
        type=float,
        default=[0.9, 0.2],
        nargs="+",
        help="min area of crops",
    )
    parser.add_argument(
        "--max_scale_crops",
        type=float,
        default=[1, 0.7],
        nargs="+",
        help="max area of crops",
    )
    parser.add_argument(
        "--constrained_cropping",
        type=str2bool,
        default=False,
        help="condition small crops on key crop",
    )
    parser.add_argument(
        "--auto_augment",
        type=int,
        default=[],
        nargs="+",
        help="Apply auto-augment 50 % of times to the selected crops",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="coefficient of the global contrastive loss",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="coefficient of the local contrastive loss",
    )

    parser.add_argument("--start_session", type=int, default=0)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="loading model parameter from a specific dir",
    )
    parser.add_argument(
        "--pre_trained_encoder_path",
        type=str,
        default="checkpoint/moco_v2_800ep_pretrain.pth.tar",
        help="loading model parameter from a specific dir",
    )

    # ABLATION AND FURTHER RESEARCH
    parser.add_argument(
        "--ce_loss_factor",
        type=float,
        default=1.0,
        help="coefficient of the cross-entropy loss",
    )
    parser.add_argument(
        "--moco_loss_factor",
        type=float,
        default=1.0,
        help="coefficient of the moco loss",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        help="encoder architecture",
    )

    # few-shot configs
    parser.add_argument(
        "--shots",
        type=int,
        default=-1,
        help="number of shots; -1 means shots taken as the defaults for the dataset",
    )
    parser.add_argument(
        "--ways",
        type=int,
        default=-1,
        help="number of ways; -1 means ways taken as the defaults for the dataset",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=-1,
        help="number of sessions; -1 means sessions taken as the defaults for the dataset",
    )

    # about training
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--incft",
        type=str2bool,
        default=False,
        help="incrmental finetuning",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=15,
        help="evaluation frequency",
    )
    parser.add_argument("--num_mlp", type=int, default=2)
    parser.add_argument("--pre_train_epochs", type=int, default=0)
    parser.add_argument("--pre_train_lr", type=float, default=0.001)

    # distributed training
    parser.add_argument(
        "--distributed",
        type=str2bool,
        default=False,
        help="distributed training",
    )
    parser.add_argument(
        "--distributed_launcher",
        type=str,
        default="pytorch",
        choices=["pytorch", "slurm"],
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )

    # FSCIT configs
    parser.add_argument("--pre_trained_url", type=str, default=None)
    parser.add_argument("--freeze_vit", type=str2bool, default=False)
    parser.add_argument("--freeze_layer_after", type=int, default=-1)
    parser.add_argument(
        "--pet_cls",
        type=str,
        default=None,
    )  # MUST BE NONE BY DEFAULT TO AVOID PRE-COMMIT ISSUES WITH PUBLIC CODES
    parser.add_argument("--adapt_blocks", type=int, default=0)
    parser.add_argument("--tune_encoder_epoch", type=int, default=0)
    parser.add_argument("--encoder_lr_factor", type=float, default=1)
    parser.add_argument("--rank", type=int, default=5)

    # FSCIT ablation configs
    parser.add_argument("--limited_base_class", type=int, default=-1)
    parser.add_argument("--limited_base_samples", type=float, default=1)

    return parser


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
