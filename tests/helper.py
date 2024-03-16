"""Helper functions for the tests."""

import argparse

from utils.train_utils import (
    get_dataset_configs,
    override_training_configs,
    sanity_check,
)


def get_default_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = {
        "adapt_blocks": [0, 1],
        "alpha": 0.5,
        "auto_augment": [],
        "base_class": 60,
        "base_mode": "ft_cos",
        "batch_size_base": 2,
        "batch_size_new": 0,
        "beta": 0.5,
        "ce_loss_factor": 1.0,
        "constrained_cropping": False,
        "dataroot": "./data",
        "dataset": "cifar100",
        "debug": False,
        "decay": 0.0005,
        "distributed": False,
        "distributed_backend": "nccl",
        "distributed_launcher": "pytorch",
        "encoder": "vit-b16",
        "encoder_lr_factor": 0.1,
        "epochs_base": 100,
        "epochs_new": 10,
        "eval_freq": 15,
        "exp_name": "cifar100_baseline",
        "encoder_ft_start_layer": 500,
        "freeze_vit": True,
        "gamma": 0.1,
        "gpu": "0",
        "incft": False,
        "limited_base_class": -1,
        "limited_base_samples": 1,
        "lr_base": 0.1,
        "lr_new": 0.1,
        "lrb": 0.1,
        "lrw": 0.1,
        "max_scale_crops": [1, 0.14],
        "milestones": [60, 80, 100],
        "min_scale_crops": [0.2, 0.05],
        "moco_dim": 128,
        "moco_k": 65536,
        "moco_loss_factor": 0.1,
        "moco_m": 0.999,
        "moco_t": 0.07,
        "model_dir": None,
        "momentum": 0.9,
        "new_mode": "avg_cos",
        "not_data_init": False,
        "num_classes": 100,
        "num_crops": [2, 1],
        "num_mlp": 2,
        "num_workers": 0,
        "pet_cls": None,
        "pre_train_epochs": 0,
        "pre_train_lr": 0.001,
        "rank": 5,
        "save_path": "checkpoint/cifar100_baseline",
        "schedule": "Step",
        "fsl_setup": "FSCIL",
        "seed": 1,
        "sessions": 11,
        "shot": 5,
        "num_views": 2,
        "hf_model_checkpoint": "google/vit-base-patch16-224-in21k",
        "start_session": 0,
        "step": 40,
        "temperature": 16,
        "test_batch_size": 4,
        "size_crops": [224, 224],
        "encoder_ft_start_epoch": 0,
        "way": 5,
        "add_bias_in_classifier": False,
        "pet_on_teacher": False,
    }

    args_ = argparse.Namespace(**args)
    args_ = get_dataset_configs(args_)
    args_ = override_training_configs(args_)
    sanity_check(args_)
    return args_


def get_lora_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.pet_cls = "LoRA"

    return args


def get_10way_10shot_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.fsl_setup = "FSCIT"
    args.shot = 10
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args


def get_food101_dataset_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.dataset = "food101"

    args.fsl_setup = "FSCIT"
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args


def get_caltech101_dataset_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.dataset = "caltech101"

    args.fsl_setup = "FSCIT"
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args


def get_country211_dataset_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.dataset = "country211"

    args.fsl_setup = "FSCIT"
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args


def get_eurosat_dataset_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.dataset = "eurosat"

    args.fsl_setup = "FSCIT"
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args


def get_fgvc_aircraft_dataset_args() -> argparse.Namespace:
    """Get default arguments for the tests."""
    args = get_default_args()
    args.dataset = "fgvc_aircraft"

    args.fsl_setup = "FSCIT"
    args = get_dataset_configs(args)
    args = override_training_configs(args)
    sanity_check(args)

    return args
