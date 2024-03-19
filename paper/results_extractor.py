"""Script for extracting resullts."""

from typing import List, Optional

import pandas as pd

from utils.constants import num_classes


dataset_name_acronym = {
    "caltech101": "CAL101",
    "cifar100": "CIF100",
    "country211": "COU211",
    "cub200": "CUB200",
    "dtd": "DTD",
    "eurosat": "EUSAT",
    "fgvc_aircraft": "AirCr.",
    "food101": "Food101",
    "gtsrb": "GTSRB",
    "mini_imagenet": "MiniIN",
    "oxford_flowers": "FLO102",
    "oxford_pets": "Pets",
    "resisc45": "RES45",
    "stanford_cars": "Cars",
    "sun397": "SUN397",
    "voc2007": "VOC",
}


def load_tsv(file_path: str) -> pd.DataFrame:
    """Load a tsv file and return a pandas dataframe."""
    # load the tsv file
    return pd.read_csv(file_path, sep="\t")


def load_results(
    exp_suffix: str,
    search_key: str,
    datsets: Optional[List] = None,
    result_keys: List[str] = ["all_last"],  # noqa B006
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
    datasets = datsets or sorted(num_classes.keys())
    results = {}  # type: ignore
    for dataset in datasets:
        results[dataset] = {}
        file_path = f"results/{dataset}_{exp_suffix}.tsv"
        result = load_tsv(file_path)
        groups = result.groupby(search_key)
        for name, group in groups:
            results[dataset][name] = {}
            for result_key in result_keys:
                results[dataset][name][result_key] = group[result_key].max()

    return results
