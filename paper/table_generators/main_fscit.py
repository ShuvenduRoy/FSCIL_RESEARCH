"""Main Table: Baseline all (Table 3)."""

from paper.results_extractor import load_results
from utils.constants import encoder_name_acronym


def main_fscit() -> None:
    """Generate naive baseline performance for all encoders."""
    with open("paper/tables/main_fscit.tex", "r") as f:
        lines = f.readlines()
        result_lines = lines[:6]

    results_prototype = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )
    all_datasets = sorted(results_prototype.keys())

    encoder = "google/vit-base-patch16-224-in21k"
    line = "Pro. tun. & " + encoder_name_acronym[encoder][4:]
    total = 0
    for dataset in all_datasets:
        line += f" & {results_prototype[dataset][encoder]['all_last']:.2f}"
        total += results_prototype[dataset][encoder]["all_last"]
    line += f" & {round(total/len(all_datasets), 2)} \\\\\n"
    result_lines.append(line)

    encoder = "google/vit-large-patch16-384"
    line = "Pro. tun. & " + encoder_name_acronym[encoder][4:]
    total = 0
    for dataset in all_datasets:
        line += f" & {results_prototype[dataset][encoder]['all_last']:.2f}"
        total += results_prototype[dataset][encoder]["all_last"]
    line += f" & {round(total/len(all_datasets), 2)} \\\\\n"
    result_lines.append(line)

    result_lines.append("\\bottomrule\n")
    result_lines.append("\\end{tabular}\n")
    result_lines.append("}\n")

    with open("paper/tables/main_fscit.tex", "w") as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    main_fscit()
