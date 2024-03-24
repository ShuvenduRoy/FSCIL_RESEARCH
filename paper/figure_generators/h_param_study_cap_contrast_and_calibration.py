"""H.param analysis."""

import matplotlib.pyplot as plt
import numpy as np

from paper.results_extractor import load_results, rotate_dict
from utils.constants import colors, dataset_name_acronym


def h_param_study_cap_contrast_and_calibration() -> None:  # noqa: PLR0915, PLR0912
    """Generate results for n shot."""
    # bar plot
    fig, axs = plt.subplots(4, 1, figsize=(16, 18))

    results = load_results(
        "FSCIT_contrast_and_calibration_3",
        search_key="lr_base",
        result_keys=["all_last"],
    )
    for key in results:
        for item in results[key]:
            results[key][item] = results[key][item]["all_last"]

    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = round(
            sum(
                [results[dataset][shot] for dataset in all_datasets],
            )
            / len(all_datasets),
            2,
        )
    all_datasets.append("Avg")

    bar_results = rotate_dict(results)
    x = np.arange(len(all_datasets))  # the label locations
    width = 1 / (len(bar_results) + 1)

    for multiplier, (attribute, measurement) in enumerate(bar_results.items()):
        offset = width * multiplier
        rects = axs[0].bar(
            x + offset,
            measurement.values(),
            width,
            label=attribute,
            color=colors[multiplier],
        )
        axs[0].bar_label(rects, padding=3, label_type="edge", rotation=60)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_ylabel("Average Acc.")
    axs[0].set_xticks(
        x + width,
        [dataset_name_acronym.get(dataset, dataset) for dataset in all_datasets],
        rotation=35,
    )
    axs[0].legend(loc="upper left", ncols=3)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)

    # ADAPT LAYERS PLOT
    sub_plot = 1
    results = load_results(
        "FSCIT_contrast_and_calibration_3",
        search_key="encoder_lr_factor",
        result_keys=["all_last"],
    )
    for key in results:
        for item in results[key]:
            results[key][item] = results[key][item]["all_last"]

    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = round(
            sum(
                [results[dataset][shot] for dataset in all_datasets],
            )
            / len(all_datasets),
            2,
        )
    all_datasets.append("Avg")

    bar_results = rotate_dict(results)
    x = np.arange(len(all_datasets))  # the label locations
    width = 1 / (len(bar_results) + 1)

    for multiplier, (attribute, measurement) in enumerate(bar_results.items()):
        offset = width * multiplier
        rects = axs[sub_plot].bar(
            x + offset,
            measurement.values(),
            width,
            label=attribute,
            color=colors[multiplier],
        )
        axs[sub_plot].bar_label(rects, padding=3, label_type="edge", rotation=60)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[sub_plot].set_ylabel("Average Acc.")
    axs[sub_plot].set_xticks(
        x + width,
        [dataset_name_acronym.get(dataset, dataset) for dataset in all_datasets],
        rotation=35,
    )
    axs[sub_plot].legend(loc="upper left", ncols=3)
    axs[sub_plot].spines["top"].set_visible(False)
    axs[sub_plot].spines["right"].set_visible(False)

    # LR_BASE PLOT
    sub_plot = 2
    results = load_results(
        "FSCIT_contrast_and_calibration_3",
        search_key="encoder_ft_start_layer",
        result_keys=["all_last"],
    )
    for key in results:
        for item in results[key]:
            results[key][item] = results[key][item]["all_last"]

    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = round(
            sum(
                [results[dataset][shot] for dataset in all_datasets],
            )
            / len(all_datasets),
            2,
        )
    all_datasets.append("Avg")

    bar_results = rotate_dict(results)

    x = np.arange(len(all_datasets))  # the label locations
    width = 1 / (len(bar_results) + 1)

    for multiplier, (attribute, measurement) in enumerate(bar_results.items()):
        offset = width * multiplier
        rects = axs[sub_plot].bar(
            x + offset,
            measurement.values(),
            width,
            label=attribute,
            color=colors[multiplier],
        )
        axs[sub_plot].bar_label(rects, padding=3, label_type="edge", rotation=60)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[sub_plot].set_ylabel("Average Acc.")
    axs[sub_plot].set_xticks(
        x + width,
        [dataset_name_acronym.get(dataset, dataset) for dataset in all_datasets],
        rotation=35,
    )
    axs[sub_plot].legend(loc="upper left", ncols=3)
    axs[sub_plot].spines["top"].set_visible(False)
    axs[sub_plot].spines["right"].set_visible(False)

    # LR_BASE PLOT
    sub_plot = 3
    results = load_results(
        "FSCIT_contrast_and_calibration_3",
        search_key="encoder_ft_start_epoch",
        result_keys=["all_last"],
    )
    for key in results:
        for item in results[key]:
            results[key][item] = results[key][item]["all_last"]

    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = round(
            sum(
                [results[dataset][shot] for dataset in all_datasets],
            )
            / len(all_datasets),
            2,
        )
    all_datasets.append("Avg")

    bar_results = rotate_dict(results)
    x = np.arange(len(all_datasets))  # the label locations
    width = 1 / (len(bar_results) + 1)

    for multiplier, (attribute, measurement) in enumerate(bar_results.items()):
        offset = width * multiplier
        rects = axs[sub_plot].bar(
            x + offset,
            measurement.values(),
            width,
            label=attribute,
            color=colors[multiplier],
        )
        axs[sub_plot].bar_label(rects, padding=3, label_type="edge", rotation=60)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[sub_plot].set_ylabel("Average Acc.")
    axs[sub_plot].set_xticks(
        x + width,
        [dataset_name_acronym.get(dataset, dataset) for dataset in all_datasets],
        rotation=35,
    )
    axs[sub_plot].legend(loc="upper left", ncols=3)
    axs[sub_plot].spines["top"].set_visible(False)
    axs[sub_plot].spines["right"].set_visible(False)

    plt.savefig("paper/figures/h_param_study_cap_contrast_and_calibration.pdf")


if __name__ == "__main__":
    h_param_study_cap_contrast_and_calibration()
    plt.show()
