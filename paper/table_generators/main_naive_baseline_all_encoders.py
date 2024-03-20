"""Main Table: Naive Baseline Performance for All Encoders."""

from paper.results_extractor import load_results
from utils.constants import encoder_name_acronym, encoder_params


def main_generate_naive_baseline_all_encoders() -> None:
    """Generate naive baseline performance for all encoders."""
    results = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
    )
    all_datasets = sorted(results.keys())
    all_encoders = sorted(
        results["caltech101"].keys(),
        key=lambda x: list(encoder_name_acronym.keys()).index(x),
    )

    results["Avg."] = {}
    # calculate avg across datasets
    for encoder in all_encoders:
        total = 0
        results["Avg."][encoder] = {}
        for dataset in all_datasets:
            total += results[dataset][encoder]["all_last"]
        results["Avg."][encoder]["all_last"] = total / len(all_datasets)

    with open("paper/tables/naive_baseline_main.tex", "r") as f:
        lines = f.readlines()
    result_lines = lines[:3]

    for encoder in all_encoders:
        line = encoder_name_acronym[encoder]  # "google/vit-base-patch16-224"
        line += " & " + str(encoder_params[encoder]) + "M"
        line += f" & {results['Avg.'][encoder]['all_last']:.2f}"
        result_lines.append(line + "\\\\\n")

    result_lines.append("\\bottomrule\n")
    result_lines.append("\\end{tabular}\n")

    with open("paper/tables/naive_baseline_main.tex", "w") as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    main_generate_naive_baseline_all_encoders()
