"""The main training file."""

import os
from pprint import pprint

from methods.fscil import FSCILTrainer
from utils import get_command_line_parser, get_dataset_configs, set_seed


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = get_dataset_configs(args)

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

    trainer = FSCILTrainer(args)

    trainer.train()
