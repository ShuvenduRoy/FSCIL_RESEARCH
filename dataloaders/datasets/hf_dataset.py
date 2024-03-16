"""Load food101 dataset."""

import argparse
from typing import Any

from datasets import load_dataset

from dataloaders.utils import (
    get_few_shot_samples_indices_per_class,
    get_session_classes,
)


def hf_dataset_name_map(dataset_name: str) -> str:
    """Map dataset name to Hugging Face dataset name."""
    return {
        "food101": "food101",
        "caltech101": "clip-benchmark/wds_vtab-caltech101",
        "country211": "clip-benchmark/wds_country211",
        "eurosat": "clip-benchmark/wds_vtab-eurosat",
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "omniglot": "omniglot",
        "aircraft": "aircraft",
        "dtd": "dtd",
        "vgg-flowers": "vgg_flowers",
        "stanford-cars": "stanford_cars",
        "svhn": "svhn_cropped",
        "ucf101": "ucf101",
        "imagenet2012": "imagenet2012",
    }[dataset_name]


def get_hf_data(dataset_name: str, split: str) -> Any:
    """Get data from Hugging Face dataset."""
    dataset = load_dataset(hf_dataset_name_map(dataset_name))
    if split == "validation" and split not in dataset:
        if "test" in dataset:
            split = "test"
        elif "valid" in dataset:
            split = "valid"
    dataset = dataset[split]

    if dataset_name in ["caltech101", "eurosat"]:
        dataset = dataset.remove_columns(["__key__", "__url__"])
        dataset = dataset.rename_column("cls", "label")
        dataset = dataset.rename_column("webp", "image")

    if dataset_name == "country211":
        dataset = dataset.remove_columns(["__key__", "__url__"])
        dataset = dataset.rename_column("cls", "label")
        dataset = dataset.rename_column("jpg", "image")

    return dataset


def hf_dataset(
    root: str,
    args: argparse.Namespace,
    train: bool = True,
    download: bool = False,
    session: int = 0,
    transformations: Any = None,
) -> Any:
    """Initialize HF dataset.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    """
    split = "train" if train else "validation"
    dataset = get_hf_data(args.dataset, split)

    if session == -1:  # way of getting the whole dataset (not needed for this project)
        return dataset

    classes_at_current_session = get_session_classes(args, session)

    if train:  # few-shot training samples
        sample_ids = get_few_shot_samples_indices_per_class(
            args.dataset,
            classes_at_current_session,
            args.shot,
        )
    else:  # validation; all samples of the classes at curr session
        sample_ids = [
            i
            for i, label in enumerate(dataset["label"])
            if label in classes_at_current_session
        ]
    dataset = dataset.select(sample_ids)

    def multi_view_transform(image: Any) -> Any:
        """Apply multi-view transformation."""
        return [transformations(image) for _ in range(args.num_views)]

    def preprocess_train(example_batch: Any) -> Any:
        """Apply train_transforms across a batch."""
        example_batch["image"] = [
            multi_view_transform(image.convert("RGB"))
            for image in example_batch["image"]
        ]
        return example_batch

    dataset.set_transform(preprocess_train)

    return dataset
