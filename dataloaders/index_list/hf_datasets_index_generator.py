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


# country211
dataset_name = "country211"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_country211")

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


# country211
dataset_name = "eurosat"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-eurosat")

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


# country211
dataset_name = "fgvc_aircraft"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_fgvc_aircraft")

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


# gtsrb
dataset_name = "gtsrb"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_gtsrb")

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

# oxford_flowers
dataset_name = "oxford_flowers"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("HuggingFaceM4/Oxford-102-Flower")

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

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

# oxford_flowers
dataset_name = "oxford_pets"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-pets")

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

# resisc45
dataset_name = "resisc45"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_vtab-resisc45")

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

# resisc45
dataset_name = "stanford_cars"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_cars")

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


# voc2007
dataset_name = "voc2007"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset("clip-benchmark/wds_voc2007")

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

# dtd
dataset_name = "dtd"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(
        "HuggingFaceM4/DTD_Describable-Textures-Dataset",
        "partition_1",
    )

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

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

# objectnet
dataset_name = "objectnet"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(
        "clip-benchmark/wds_objectnet",
    )

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

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


# sun397
dataset_name = "sun397"
if not os.path.exists(f"dataloaders/index_list/{dataset_name}_index_list.txt"):
    dataset = load_dataset(
        "clip-benchmark/wds_sun397",
    )

    num_classes = len(set(dataset["train"]["label"]))
    print("Total classes", num_classes)

    samples_per_class = 32
    class_counts = class_count(dataset["train"]["label"])
    samples_per_class = min(*class_counts.values(), samples_per_class)
    print("Samples per class", samples_per_class)

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
