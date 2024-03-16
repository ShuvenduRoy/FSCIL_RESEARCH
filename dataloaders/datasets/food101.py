"""Load food101 dataset."""

import argparse
from typing import Any

from datasets import load_dataset

from dataloaders.utils import (
    get_few_shot_samples_indices_per_class,
    get_session_classes,
)


def food101dataset(
    root: str,
    args: argparse.Namespace,
    train: bool = True,
    download: bool = False,
    session: int = 0,
    transformations: Any = None,
) -> Any:
    """Initialize Food101 dataset.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    """
    split = "train" if train else "validation"
    dataset = load_dataset("food101")[split]
    if session == -1:  # way of getting the whole dataset (not needed for this project)
        return dataset

    classes_at_current_session = get_session_classes(args, session)

    if train:  # few-shot training samples
        sample_ids = get_few_shot_samples_indices_per_class(
            "food101",
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
