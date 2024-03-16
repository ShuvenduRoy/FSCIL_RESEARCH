"""Generate few-shot index list for Food101 dataset."""

import os
import random

from datasets import load_dataset


def class_count(labels: list[int]) -> dict[int, int]:
    """Count the number of samples per class."""
    classes = set(labels)
    counts = {c: labels.count(c) for c in classes}
    print("Class counts", counts)
    print("Min class count", min(counts.values()))
    return counts


# Food101
dataset_name = "food101"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(dataset_name)

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i
            for i, label in enumerate(dataset["train"]["label"])
            if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")


# Caltech101
dataset_name = "caltech101"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-caltech101")

    num_classes = len(set(dataset["train"]["cls"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["cls"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

    selected_samples = {}
    for class_index in range(num_classes):
        indices = [
            i for i, label in enumerate(dataset["train"]["cls"]) if label == class_index
        ]
        selected_samples[class_index] = random.sample(indices, samples_per_class)

    with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
        for class_index, indices in selected_samples.items():
            for index in indices:
                file.write(f"{class_index} {index}\n")
