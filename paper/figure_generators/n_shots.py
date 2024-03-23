"""Results generator for different number of shots."""

import matplotlib.pyplot as plt

from paper.results_extractor import load_results
from utils.constants import colors, dataset_name_acronym


def generate_n_shots_figures() -> None:  # noqa: PLR0915
    """Generate results for n shot."""
    results = load_results(
        "FSCIT_prototype_fscit_different_shots_3",
        search_key="shot",
        result_keys=["all_last", "all"],
        filter_key=["hf_model_checkpoint"],
        filter_value=["google/vit-base-patch16-224-in21k"],
    )

    # average over datasets
    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = sum(
            [results[dataset][shot]["all_last"] for dataset in all_datasets],
        ) / len(all_datasets)

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    fontsize = 18
    ax.plot(
        results["Avg"].keys(),
        results["Avg"].values(),
        label="Prototype learning",
        color=colors[1],
        linewidth=3,
    )

    ax.set_xlabel("Shots", fontsize=fontsize + 3)
    ax.set_ylabel("Average accuracy", fontsize=fontsize + 3)

    ax.legend()

    ax.grid(color="lightgray", linestyle="--", linewidth=0.8)
    ax.get_legend().get_frame().set_linewidth(0.0)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize + 1)

    plt.savefig("paper/figures/n_shots_main_b21k.pdf")

    # APPENDIX TABLE
    fig, axs = plt.subplots(4, 4, figsize=(16, 18))
    for i, dataset in enumerate(all_datasets):
        col = i % 4
        row = i // 4
        if col == 0 and row == 0:
            axs[row, col].plot(
                results[dataset].keys(),
                [results[dataset][shot]["all_last"] for shot in all_shots],
                label="Prototype learning",
            )
        else:
            axs[row, col].plot(
                results[dataset].keys(),
                [results[dataset][shot]["all_last"] for shot in all_shots],
            )
        axs[row, col].grid(color="lightgray", linestyle="--", linewidth=0.8)
        axs[row, col].set_title(f"{dataset_name_acronym[dataset]}")

    fig.legend(loc="lower center", ncol=4, fontsize=fontsize - 3)
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.1)
    plt.savefig("paper/figures/n_shots_appendix_b21k.pdf")

    # MAIN L386
    results = load_results(
        "FSCIT_prototype_fscit_different_shots_3",
        search_key="shot",
        result_keys=["all_last", "all"],
        filter_key=["hf_model_checkpoint"],
        filter_value=["google/vit-large-patch16-384"],
    )

    # average over datasets
    all_datasets = sorted(results.keys())
    all_shots = sorted(results[all_datasets[0]].keys())
    results["Avg"] = {}
    for shot in all_shots:
        results["Avg"][shot] = sum(
            [results[dataset][shot]["all_last"] for dataset in all_datasets],
        ) / len(all_datasets)

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    fontsize = 18
    ax.plot(
        results["Avg"].keys(),
        results["Avg"].values(),
        label="Prototype learning",
        color=colors[1],
        linewidth=3,
    )

    ax.set_xlabel("Shots", fontsize=fontsize + 3)
    ax.set_ylabel("Average accuracy", fontsize=fontsize + 3)

    ax.legend()

    ax.grid(color="lightgray", linestyle="--", linewidth=0.8)
    ax.get_legend().get_frame().set_linewidth(0.0)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize + 1)

    plt.savefig("paper/figures/n_shots_main_l386.pdf")

    # APPENDIX TABLE L386
    fig, axs = plt.subplots(4, 4, figsize=(16, 18))
    for i, dataset in enumerate(all_datasets):
        col = i % 4
        row = i // 4
        if col == 0 and row == 0:
            axs[row, col].plot(
                results[dataset].keys(),
                [results[dataset][shot]["all_last"] for shot in all_shots],
                label="Prototype learning",
            )
        else:
            axs[row, col].plot(
                results[dataset].keys(),
                [results[dataset][shot]["all_last"] for shot in all_shots],
            )
        axs[row, col].grid(color="lightgray", linestyle="--", linewidth=0.8)
        axs[row, col].set_title(f"{dataset_name_acronym[dataset]}")

    fig.legend(loc="lower center", ncol=4, fontsize=fontsize - 3)
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.1)
    plt.savefig("paper/figures/n_shots_appendix_l386.pdf")


if __name__ == "__main__":
    generate_n_shots_figures()
