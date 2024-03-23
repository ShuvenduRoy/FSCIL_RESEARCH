"""Script for extracting resullts."""

from typing import Any, List, Optional

import pandas as pd

from utils.constants import num_classes


def rotate_dict(d: dict) -> dict:
    """Rotate a dictionary."""
    new_dict: dict = {}
    for key, value in d.items():
        for sub_key, sub_value in value.items():
            new_dict.setdefault(sub_key, {})[key] = sub_value

    return new_dict


def load_tsv(file_path: str) -> pd.DataFrame:
    """Load a tsv file and return a pandas dataframe."""
    # load the tsv file
    return pd.read_csv(file_path, sep="\t")


def load_results(
    exp_suffix: str,
    search_key: str,
    datasets: Optional[List] = None,
    result_keys: List[str] = ["all_last"],  # noqa B006
    filter_key: Optional[List[Any]] = None,
    filter_value: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Load results.

    Parameters
    ----------
    exp_suffix : str
        The suffix of the experiment will be added
        to the dataset name to load the results.
    search_key : str
        Criteria for partitioning results.
    datasets : List, optional
        The datasets for which the results will be loaded.
        if None, then all the datasets will be loaded.

    """
    datasets = datasets or sorted(num_classes.keys())
    results = {}  # type: ignore
    for dataset in datasets:
        results[dataset] = {}
        file_path = f"results/{dataset}_{exp_suffix}.tsv"
        result = load_tsv(file_path)
        if filter_key is not None:
            for key, value in zip(filter_key, filter_value):  # type: ignore
                result = result[result[key] == value]
        groups = result.groupby(search_key)
        for name, group in groups:
            results[dataset][name] = {}
            select_id = group["all_last"].idxmax()
            for result_key in result_keys:
                results[dataset][name][result_key] = group[result_key][select_id]

    return results
