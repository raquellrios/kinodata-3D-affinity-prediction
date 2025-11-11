from pathlib import Path
from functools import singledispatchmethod
from typing import List, Optional, Protocol
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold, KFold
import pandas as pd

from kinodata.data.data_split import Split
from kinodata.data.utils.cluster import AffinityPropagation
from kinodata.data.utils.similarity import BLOSUMSubstitutionSimilarity
from kinodata.data.dataset import KinodataDocked
from kinodata.data.dataset_davids_data import DavidsdataDocked
from collections import Counter

def _print_scaffold_stats_per_fold(scaffolds: np.ndarray,
                                   splits: List[Split],
                                   tag: str = "") -> None:
    """
    Print how many (unique) scaffolds and molecules each part
    (train / val / test) of every fold contains.
    """
    for fold_id, sp in enumerate(splits):
        print(f"\n[{tag}] fold {fold_id}")
        for name, idx in zip(["train", "val", "test"],
                             [sp.train_split, sp.val_split, sp.test_split]):
            part_scaffolds = scaffolds[idx]
            cnt = Counter(part_scaffolds)
            print(f"  {name:5s}: {len(cnt):4d} scaffolds, "
                  f"{len(part_scaffolds):5d} mols "
                  f"(largest={cnt.most_common(1)[0][1]})")


def _split_random(a: np.ndarray, percentile: float, seed: int = 0):
    rng = default_rng(seed)
    pivot_ = int(a.shape[0] * percentile)
    permuted = rng.permutation(a)
    return permuted[:pivot_], permuted[pivot_:]


def _generator_to_list(generator):
    splits = [
        Split(train_index, *_split_random(test_index, 0.5))  # type: ignore
        for train_index, test_index in generator
    ]
    return splits

def limit_scaffold_representation(group_index, max_samples_per_scaffold):
    """
    Limits the number of samples per scaffold to ensure better split distribution.

    Args:
        group_index (np.ndarray): Array of scaffold identifiers.
        max_samples_per_scaffold (int): Maximum allowed samples per scaffold.

    Returns:
        np.ndarray: Filtered indices after limiting scaffold representation.
    """
    scaffold_counts = pd.DataFrame({'scaffold': group_index}).value_counts()

    #print("the scaffolds counts inside the limit_scaffolds_rep")
    #print(scaffold_counts)

    filtered_indices = []
    
    for scaffold, count in scaffold_counts.items():
        #print("the count is "+str(count))
        scaffold_indices = np.where(group_index == scaffold)[0]
        #print("the len of the scaffold indices is "+str(len(scaffold_indices)))
        limited_indices = scaffold_indices[:max_samples_per_scaffold]  # Limit to max_samples_per_scaffold
        #print("the len of the limited indices are "+str(len(limited_indices)))
        #print(limited_indices[:5])
        filtered_indices.extend(limited_indices)
    
    return np.array(filtered_indices)


#def group_k_fold_split(
#    group_index: np.ndarray,
#    k: int,
#) -> List[Split]:
#    
#    print("inside the group_k_fold_split")
#    print("group index is ")
#    print(group_index)
#    print("the len of group index is " +str(group_index.shape[0]))
#    group_k_fold = GroupKFold(k)
#    _X = np.zeros((group_index.shape[0], 1))
#    generator = group_k_fold.split(_X, groups=group_index)
#    return _generator_to_list(generator)

def group_k_fold_split(
    group_index: np.ndarray,
    k: int,
    max_samples_per_scaffold: Optional[int] = None,
) -> List[Split]:
    """
    Custom GroupKFold split with optional filtering of overrepresented scaffolds.

    Args:
        group_index (np.ndarray): Array of scaffold identifiers.
        k (int): Number of folds.
        max_samples_per_scaffold (int, optional): Max samples per scaffold. Default is None.

    Returns:
        List[Split]: Splits for training, validation, and test.
    """
    #print("group indices before filtered")
    #print(group_index[:5])
    #print("group index shape before filtering "+str(np.shape(group_index)))
    if max_samples_per_scaffold is not None:
        filtered_indices = limit_scaffold_representation(group_index, max_samples_per_scaffold)
        group_index = group_index[filtered_indices]
        #print("group indices after filtered")
        #print(group_index[:5])
        #print("group index shape after filtering "+str(np.shape(group_index)))
   
    group_k_fold = GroupKFold(k)
    _X = np.zeros((group_index.shape[0], 1))
    generator = group_k_fold.split(_X, groups=group_index)
    #return _generator_to_list(generator)
    splits = []
    for train_idx, test_idx in generator:
        train_global = filtered_indices[train_idx]
        test_global = filtered_indices[test_idx]

        # Further split the test set into validation and test
        val_idx, test_idx = _split_random(test_global, 0.5)  # Adjust as needed
        splits.append(Split(train_global, val_idx, test_idx))

    return splits



def random_k_fold_split(data_index: np.ndarray, k: int) -> List[Split]:
    k_fold = KFold(k, shuffle=True)
    generator = k_fold.split(data_index)
    return _generator_to_list(generator)


class KinodataKFoldSplit:
    """
    Wraps all supported splitting methods.
    and implements dataset-dependent caching.
    Splits are cached in a datasets processed dir.
    """

    pocket_clustering = AffinityPropagation()
    pocket_similarity_measure = BLOSUMSubstitutionSimilarity


    #print("running KinodataKfoldSplit")

    def __init__(self, split_type: str, k: int, max_samples_per_scaffold: Optional[int] = None) -> None:
        assert split_type in (
            "scaffold-k-fold",
            "pocket-k-fold",
            "random-k-fold",
        ), f"Unknown split type {split_type}"
        self.k = k
        self.split_type = split_type
        self.max_samples_per_scaffold = max_samples_per_scaffold

    @singledispatchmethod
    def cache_dir(self, dataset) -> Path:
        raise NotImplementedError(f"cahe_dir({type(dataset)})")

    @cache_dir.register
    def _(self, dataset: KinodataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type
    
    @cache_dir.register
    def _(self, dataset: DavidsdataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type

    @cache_dir.register
    def _(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.split_type

    def split_files(
        self,
        dataset: Optional[KinodataDocked] = None,
        dataset_dir: Optional[Path] = None,
    ) -> List[Path]:
        if dataset_dir is not None:
            cache_dir = self.cache_dir(dataset_dir)
        if dataset is not None:
            cache_dir = self.cache_dir(dataset)
        assert cache_dir is not None
        return [cache_dir / f"{i}:{self.k}.csv" for i in range(1, self.k + 1)] #here it is where the csv is loaded!

    #def _split(self, dataset: KinodataDocked):
    def _split(self, dataset):
        if self.split_type == "scaffold-k-fold":
            scaffolds, idents = zip(*[(data.scaffold, data.ident) for data in dataset])
            scaffolds = np.array(scaffolds)
            splits = group_k_fold_split(group_index=scaffolds, k=self.k, max_samples_per_scaffold=1000) #=self.max_samples_per_scaffold)

            # Debugging: Check scaffold distribution
            scaffold_counts = pd.DataFrame({'scaffold': scaffolds}).value_counts()
            print(f"Scaffold distribution scaffold for {self.k} split:")
            print(scaffold_counts)
            _print_scaffold_stats_per_fold(scaffolds, splits, tag="scaffold-k-fold")
            return splits

        if self.split_type == "pocket-k-fold":
            pocket_data = pd.DataFrame(
                {
                    "index": np.arange(len(dataset.data.pocket_sequence)),
                    "pocket_sequence": dataset.data.pocket_sequence,
                }
            )
            df_cluster_labels = self.pocket_clustering(
                pocket_data,
                "pocket_sequence",
                fn_similarity=self.pocket_similarity_measure(),
            ).sort_values(by="index", ascending=True)
            assert df_cluster_labels.shape[0] == pocket_data.shape[0]
            return group_k_fold_split(
                np.array(df_cluster_labels[self.pocket_clustering.cluster_key].values),
                k=self.k,
            )
        if self.split_type == "random-k-fold":
            idents = [data.ident for data in dataset]
            return random_k_fold_split(idents, self.k)

        #def split(self, dataset: KinodataDocked) -> List[Split]:
    def split(self, dataset) -> List[Split]:

        #print("running from here in the KInodataKfoldSplit")
        split_files = self.split_files(dataset)
        if all(f.exists() for f in split_files):
            return [Split.from_csv(f) for f in split_files]

        splits: List[Split] = self._split(dataset)

        for split, f in zip(splits, split_files):
            if not f.parents[0].exists():
                f.parents[0].mkdir()
            split.to_data_frame().to_csv(f, index=False)
            split.source_file = str(f)

        return splits
    
