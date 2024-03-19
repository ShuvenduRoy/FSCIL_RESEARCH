"""Appendix Table: Naive Baseline Performance for All Encoders."""

import matplotlib.pyplot as plt

from paper.results_extractor import dataset_name_acronym, load_results


def generate_naive_baseline_all_encoders() -> None:
    """Generate naive baseline performance for all encoders."""
    fontsize = 18

    results = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
        result_keys=["all"],
    )
    all_datasets = sorted(results.keys())
    all_encoders = sorted(results["caltech101"].keys())
    dataset_shot_names = [dataset_name_acronym[dataset] for dataset in all_datasets]

    # generate 16x16 plots for 16 datasets
    fig, axs = plt.subplots(4, 4, figsize=(16, 18))

    for i, dataset in enumerate(all_datasets):
        col = i % 4
        row = i // 4
        for _, encoder in enumerate(all_encoders):
            if encoder in results[dataset]:
                accs = results[dataset][encoder]["all"][1:-1]
                accs = [float(acc) for acc in accs.split(", ")]
                if i == 0:
                    axs[row, col].plot(accs, label=encoder.split("/")[1])
                else:
                    axs[row, col].plot(accs)
                axs[row, col].set_title(f"{dataset_shot_names[i]}")
                axs[row, col].grid(True)
    fig.legend(loc="lower center", ncol=4, fontsize=fontsize - 3)
    plt.tight_layout()

    fig.subplots_adjust(bottom=0.1)

    plt.show()


generate_naive_baseline_all_encoders()
