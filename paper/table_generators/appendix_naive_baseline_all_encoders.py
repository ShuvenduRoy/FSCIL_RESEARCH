"""Appendix Table: Naive Baseline Performance for All Encoders."""

from paper.results_extractor import load_results


results = load_results(
    "facit_baseline",
    search_key="hf_model_checkpoint",
    datsets=["caltech101", "cifar100"],
)
all_datasets = results.keys()
all_encoders = results["caltech101"].keys()

with open("paper/tables/naive_baseline_all_encoders.tex", "r") as f:
    lines = f.readlines()
result_lines = lines[:2]
result_lines.append("Encoders & " + " & ".join(all_datasets) + " \\\\ \\hline  \n")

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
