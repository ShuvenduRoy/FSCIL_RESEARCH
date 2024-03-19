"""Generates plot with base and novel accuracy breakdown for FSCIT."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_fscit_acc_breakdown(accuracies: dict, methods_on_x: bool = False) -> None:
    """Generate plot with base and novel accuracy breakdown for FSCIT."""
    class_group = ["Total", "Base", "Novel"]
    if methods_on_x:
        new_accuracies: Dict[str, List[Any]] = {}
        new_class_group = []
        for class_name in class_group:
            new_accuracies[class_name] = []
        for method, acc in accuracies.items():
            for i, class_name in enumerate(class_group):
                new_accuracies[class_name].append(acc[i])
            new_class_group.append(method)
        class_group = new_class_group
        accuracies = new_accuracies

    for key in accuracies:
        accuracies[key] = [round(x, 1) for x in accuracies[key]]

    colors = [
        "#91c5da",
        "#4692c4",
        "#f2a484",
        "#ff7f0e",
        "#ffb347",
        "#ffbb78",
        "#ffebd4",
    ]
    alphas = [1] * len(colors)

    x = np.arange(len(class_group))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    fontsize = 18

    ax.scatter(
        np.array(list(range(len(accuracies["Total"])))) + 0.1,
        accuracies["Total"],
        color=colors[2],
        zorder=10,
        marker="D",
        s=70,
        label="Total",
    )
    accuracies.pop("Total")

    i = 0
    for multiplier, (attribute, measurement) in enumerate(accuracies.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            color=colors[i],
            alpha=alphas[i],
        )
        ax.bar_label(rects, padding=3, fontsize=fontsize - 6)
        i += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy (%)", fontsize=fontsize + 3)
    ax.set_xticks(x + width, class_group)

    # rotate x labels
    plt.xticks(rotation=60)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize + 1)
    plt.subplots_adjust(bottom=0.25)
    ax.legend(loc="lower right", ncols=1, fontsize=fontsize - 3)
    ax.get_legend().get_frame().set_linewidth(0.0)
    plt.savefig("paper/figures/fscit_acc_breakdown.pdf")
    plt.show()


accuracies = {
    # total, base, novel format
    "Lin.+Pro.": (64.96, 8.6, 72.01),
    "Prom.+Pro.": (66.45, 14.47, 72.95),
    "Cont. Pro.": (71.08, 77.4, 71.08),
    "Ours": (71.08, 77.4, 71.08),
}

plot_fscit_acc_breakdown(accuracies, True)
