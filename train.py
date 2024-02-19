"""The main training file."""

import argparse
import importlib
import os
from pprint import pprint

from utils import set_seed, str2bool


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

    # FSCIT configs
    parser.add_argument("--pre_trained_url", type=str, default=None)
    parser.add_argument("--freeze_vit", type=str2bool, default=False)
    parser.add_argument("--freeze_layer_after", type=int, default=-1)
    parser.add_argument("--pet_cls", type=str, default=None)
    parser.add_argument("--adapt_blocks", type=int, default=0)
    parser.add_argument("--tune_encoder_epoch", type=int, default=0)
    parser.add_argument("--encoder_lr_factor", type=float, default=1)
    parser.add_argument("--rank", type=int, default=5)

    # FSCIT ablation configs
    parser.add_argument("--limited_base_class", type=int, default=-1)
    parser.add_argument("--limited_base_samples", type=float, default=1)

    return parser


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args.exp_name = args.dataset + "_" + args.exp_name
    if args.pre_trained_url == "None":
        args.pre_trained_url = None
    if args.adapt_blocks < 0:
        args.adapt_blocks *= -1
        args.adapt_blocks = list(range(args.adapt_blocks, 12))
    else:
        args.adapt_blocks = list(range(args.adapt_blocks))
    try:
        args.gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    except KeyError:
        args.gpu = int(args.gpu)

    # handle some default settings
    if args.dataset == "cub200":
        args.size_crops = [224, 96]
        args.min_scale_crops = [0.2, 0.05]
        args.max_scale_crops = [1, 0.14]
        args.milestones = [60, 80, 100]
    elif args.dataset == "mini_imagenet":
        args.size_crops = [84, 50]
        args.min_scale_crops = [0.2, 0.05]
        args.max_scale_crops = [1, 0.14]
        args.milestones = [40, 70, 100]
    if args.encoder == "vit-16":
        args.size_crops = [224, 224]
    if os.path.exists("/home/sneha/"):
        args.num_workers = 4

    # set the seed
    set_seed(args.seed)
    pprint(vars(args))

    trainer = importlib.import_module(
        "models.%s.fscil_trainer" % (args.project),
    ).FSCILTrainer(args)

    trainer.train()
