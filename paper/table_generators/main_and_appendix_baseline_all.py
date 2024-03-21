"""Main Table: Baseline all (Table 2, S4)."""

from paper.results_extractor import load_results
from utils.constants import dataset_name_acronym


def main_generate_all_baselines() -> None:
    """Generate naive baseline performance for all encoders."""
    with open("paper/tables/baseline_all.tex", "r") as f:
        lines = f.readlines()
        result_lines = lines[:3]
    encoder = "google/vit-base-patch16-224-in21k"

    results_prototype = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )

    results_linear = load_results(
        "FSCIT_baseline_linear_3",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )

    results_lora = load_results(
        "FSCIT_baseline_lora_3",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )

    results_dict_list = [results_prototype, results_linear, results_lora]
    results_name_list = ["Prototype learning", "Linear tuning", "PEFT (LoRA)"]

    all_datasets = sorted(results_linear.keys())

    # main paper table
    for i in range(len(results_name_list)):
        acc = sum(
            [
                results_dict_list[i][dataset][encoder]["all_last"]
                for dataset in all_datasets
            ],
        ) / len(all_datasets)
        result_lines.append(f"{results_name_list[i]} & {acc:.2f} \\\\\n")

    result_lines.append("\\bottomrule\n")
    result_lines.append("\\end{tabular}\n")

    with open("paper/tables/baseline_all.tex", "w") as f:
        f.writelines(result_lines)

    # appendix table
    num_methods = 3

    with open("paper/tables/baseline_appendix_all.tex", "r") as f:
        lines = f.readlines()
        result_lines = lines[:3]
    for dataset in list(all_datasets):
        for i in range(len(results_name_list)):
            line = (
                "\\multirow{"
                + str(num_methods)
                + "}{*}{"
                + dataset_name_acronym[dataset]
                + "} & "
                if i == 0
                else " & "
            )
            line += results_name_list[i]
            acc = results_dict_list[i][dataset][encoder]["all"][1:-1].replace(
                ", ",
                " & ",
            )
            result_lines.append(f"{line} & {acc} \\\\\n")
        result_lines.append("\\midrule\n")
    result_lines[-1] = "\\bottomrule\n"
    result_lines.append("\\end{tabular}\n")

    with open("paper/tables/baseline_appendix_all.tex", "w") as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    main_generate_all_baselines()
