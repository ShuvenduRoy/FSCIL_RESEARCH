"""Dataset class for CIFAR-100 dataset."""
import argparse
import os
import os.path
import pickle
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class Cifar100Dataset(VisionDataset):
    """`CIFAR100 Dataset.

    Parameters
    ----------
    root : str
        Root directory of dataset where directory ``cifar-10-batches-py`` exists or
        will be saved to if download is set to True.
    train : bool, optional
        If True, creates dataset from training set, otherwise creates from test set.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    download : bool, optional
        If true, downloads the dataset from the internet and puts it in root directory. 
        If dataset is already downloaded, it is not downloaded again.
    session : int, optional
        Current session.
    transformations : optional
        transformations.
    args : argparse.ArgumentParser, optional
        Arguments passed to the trainer.

    Returns
    -------
    None
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


    def __init__(self, root: str,
                train: bool = True,
                download: bool = False,
                session: int=0,
                transformations:Any=None,
                args:argparse.Namespace=None):

        super(Cifar100Dataset, self).__init__(root)
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transformations
        self.args = args

        # select the class index for base classes; first 60 classes for cifar100
        class_index = np.arange(args.base_class) if session == 0 else np.arange(args.base_class + (session - 1) * args.way, args.base_class + session * args.way)

        # ABLATION SETTING: reduce the number of base classes
        if args.limited_base_class > 0:
            class_index = class_index[:args.limited_base_class]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." +
                               " You can use download=True to download it")

        downloaded_list = self.train_list if self.train else self.test_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = np.asarray(self.targets)

        if session > 0:
            txt_path = "data/index_list/".format() + args.dataset + "/session_" + str(session + 1) + ".txt"
            with open(txt_path) as f:
                sample_index = f.read().splitlines()

        if session == 0: # base session
            self.data, self.targets = self.select_by_class_index(self.data, self.targets, class_index)
        elif train:
            self.data, self.targets = self.select_by_sample_index(self.data, self.targets, sample_index)
        else:
            self.data, self.targets = self.select_by_class_index(self.data, self.targets, class_index)

        self._load_meta()

    def select_by_class_index(self,
                            data: np.ndarray,
                            targets: np.ndarray,
                            index: np.ndarray) -> Any:
        """Select all sample of the given classes.

        Unless specified by args.limited_base_samples,
        select all the sample of the classes specified
        in the variable index.

        Parameters
        ----------
        data : np.ndarray
            Data.
        targets : np.ndarray
            Targets.
        index : np.ndarray
            Index.

        Returns
        -------
        Any
        """
        data_tmp: List[np.ndarray] = []
        targets_tmp: List[np.ndarray] = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            # Select a reduced subset of base samples per class
            # if limited_base_samples < 1.0
            if self.args:
                samples_to_select = int(len(ind_cl) * self.args.limited_base_samples)
                ind_cl = np.random.choice(ind_cl, samples_to_select, replace=False)

            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def select_by_sample_index(self,
                            data: np.ndarray,
                            targets: np.ndarray,
                            index: np.ndarray) -> Any:
        """Select all by given index.

        Parameters
        ----------
        data : np.ndarray
            Data.
        targets : np.ndarray
            Targets.
        index : np.ndarray
            Index.

        Returns
        -------
        Any
        """
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list)

        return data[ind_np], targets[ind_np]

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted." +
                               " You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Any:
        """Load one sample.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # Converting to PIL Image so that it is consistent with
        # all other datasets which return a PIL Image
        img = Image.fromarray(img)
        images: List[torch.Tensor] = [self.transform(img) for i in range(self.args.num_crops[0])]

        return images, target

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def _check_integrity(self) -> bool:
        """Check integrity of files."""
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        """Download the CIFAR-100 data if it doesn't exist already."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        """Extra repr."""
        return "Split: {}".format("Train" if self.train is True else "Test")
