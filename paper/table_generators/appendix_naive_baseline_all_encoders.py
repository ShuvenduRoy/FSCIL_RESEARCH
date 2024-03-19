"""Appendix Table: Naive Baseline Performance for All Encoders."""

from paper.results_extractor import dataset_name_acronym, load_results


def generate_naive_baseline_all_encoders() -> None:
    """Generate naive baseline performance for all encoders."""
    results = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
    )
    all_datasets = sorted(results.keys())
    all_encoders = sorted(results["caltech101"].keys())
    dataset_shot_names = [dataset_name_acronym[dataset] for dataset in all_datasets]

    with open("paper/tables/naive_baseline_all_encoders.tex", "r") as f:
        lines = f.readlines()
    result_lines = lines[:2]
    result_lines.append(
        "Encoders & " + " & ".join(dataset_shot_names) + " \\\\ \\hline\n",
    )

    for encoder in all_encoders:
        line = encoder.split("/")[1]  # "google/vit-base-patch16-224"
        for dataset in all_datasets:
            if encoder in results[dataset]:  # exp might be in progress
                res = results[dataset][encoder].get("all_last", 0)
            else:
                res = 0
            line += f" & {res:.2f}"
        result_lines.append(line + "\\\\\n")

    result_lines.append("\\bottomrule\n")
    result_lines.append("\\end{tabular}\n")

    with open("paper/tables/naive_baseline_all_encoders.tex", "w") as f:
        f.writelines(result_lines)
