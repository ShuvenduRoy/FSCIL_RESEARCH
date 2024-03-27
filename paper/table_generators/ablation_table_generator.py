"""Ablation table generator. (Table 4)."""

from paper.results_extractor import load_results


def ablation_table_generator() -> None:
    """Generate naive baseline performance for all encoders."""
    encoder = "google/vit-base-patch16-224-in21k"
    results_dict_list = []
    results_name_list = []

    # baseline
    results_prototype = load_results(
        "FSCIT__baseline",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )
    results_dict_list.append(results_prototype)
    results_name_list.append("\\xmark & \\xmark & \\xmark")

    # cap contrast only
    results_cap_contrast_only = load_results(
        "FSCIT_cap_contrast_only_3",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )
    results_dict_list.append(results_cap_contrast_only)
    results_name_list.append("\\cmark & \\xmark & \\xmark")

    # contrast and calibration
    results_contrast_and_calibration = load_results(
        "FSCIT_contrast_and_calibration_3",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )
    results_dict_list.append(results_contrast_and_calibration)
    results_name_list.append("\\cmark & \\cmark & \\xmark")

    # contrast, calibration, and incft
    results_contrast_and_calibration_ft = load_results(
        "FSCIT_cap_incft_3",
        search_key="hf_model_checkpoint",
        result_keys=["all_last", "all"],
    )
    results_dict_list.append(results_contrast_and_calibration_ft)
    results_name_list.append("\\cmark & \\cmark & \\cmark")

    all_datasets = sorted(results_prototype.keys())

    # main paper table
    with open("paper/tables/main_ablation.tex", "r") as f:
        lines = f.readlines()
        result_lines = lines[:3]
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

    with open("paper/tables/main_ablation.tex", "w") as f:
        f.writelines(result_lines)

    # appendix table

    with open("paper/tables/full_ablation.tex", "r") as f:
        lines = f.readlines()
        result_lines = lines[:5]

    for i in range(len(results_name_list)):
        total_acc = sum(
            [
                results_dict_list[i][dataset][encoder]["all_last"]
                for dataset in all_datasets
            ],
        )
        line = results_name_list[i] + " & "
        for dataset in list(all_datasets):
            line += str(results_dict_list[i][dataset][encoder]["all_last"]) + " & "

        result_lines.append(f"{line} {round(total_acc/len(all_datasets), 2)} \\\\\n")

    result_lines.append("\\bottomrule\n")
    result_lines.append("\\end{tabular}\n")

    with open("paper/tables/full_ablation.tex", "w") as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    ablation_table_generator()
