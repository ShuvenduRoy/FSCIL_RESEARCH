"""Generate few-shot index list for Food101 dataset."""

import random

from datasets import load_dataset


dataset_name = "food101"
dataset = load_dataset(dataset_name)

num_classes = len(set(dataset["train"]["label"]))
print("Total classes", num_classes)


samples_per_class = 32

selected_samples = {}
for class_index in range(num_classes):
    indices = [
        i for i, label in enumerate(dataset["train"]["label"]) if label == class_index
    ]
    selected_samples[class_index] = random.sample(indices, samples_per_class)

with open(f"dataloaders/index_list/{dataset_name}_index_list.txt", "w") as file:
    for class_index, indices in selected_samples.items():
        for index in indices:
            file.write(f"{class_index} {index}\n")
