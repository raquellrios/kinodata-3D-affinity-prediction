from pathlib import Path
from functools import singledispatchmethod
from typing import List, Optional, Protocol
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold, KFold
import pandas as pd

from .data_split import Split
from .utils.cluster import AffinityPropagation
from .utils.similarity import BLOSUMSubstitutionSimilarity
from .dataset import KinodataDocked
from .dataset_davids_data import DavidsdataDocked
from collections import Counter

from itertools import islice

def _first_n(it, n=5):
    return list(islice(it, n))

def _check_scaffold_disjoint(scaffolds: np.ndarray, sp: Split) -> dict:
    """Return sizes of scaffold overlaps between splits (should all be zero)."""
    train_set = set(scaffolds[sp.train_split])
    val_set   = set(scaffolds[sp.val_split])  if sp.val_split is not None else set()
    test_set  = set(scaffolds[sp.test_split]) if sp.test_split is not None else set()

    leak_tv = train_set & val_set
    leak_tt = train_set & test_set
    #leak_vt = val_set  & test_set

    return {
        "train∩val": (len(leak_tv), _first_n(leak_tv)),
        "train∩test": (len(leak_tt), _first_n(leak_tt)),
        #"val∩test": (len(leak_vt), _first_n(leak_vt)),
    }

def _assert_no_scaffold_leakage(scaffolds: np.ndarray, splits: List[Split], tag: str):
    for i, sp in enumerate(splits):
        res = _check_scaffold_disjoint(scaffolds, sp)
        msg_lines = [f"[{tag}] fold {i} scaffold overlaps:"]
        ok = True
        for k, (n, examples) in res.items():
            msg_lines.append(f"  {k}: {n}" + (f" (e.g. {examples})" if n else ""))
            if n > 0:
                ok = False
        if not ok:
            # You can change to `warnings.warn("\n".join(msg_lines))` if you prefer non-fatal
            raise RuntimeError("\n".join(msg_lines))
        else:
            print("\n".join(msg_lines))

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
            splits = group_k_fold_split(group_index=scaffolds, k=self.k, max_samples_per_scaffold=3000) #=self.max_samples_per_scaffold)

            # Debugging: Check scaffold distribution
            scaffold_counts = pd.DataFrame({'scaffold': scaffolds}).value_counts()
            print(f"Scaffold distribution scaffold for {self.k} split:")
            print(scaffold_counts)
            _print_scaffold_stats_per_fold(scaffolds, splits, tag="scaffold-k-fold")

            _assert_no_scaffold_leakage(scaffolds, splits, tag="scaffold-k-fold")
            
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
    

def _scaffold_array(dataset):
    """Return np.ndarray of scaffolds (aligned with dataset indexing)."""
    return np.array([d.scaffold for d in dataset])


def _index_limit_by_scaffold(scaffolds: np.ndarray, max_samples_per_scaffold: Optional[int]):
    """
    Return an index array that (optionally) limits per-scaffold sample count.
    Preserves original order within each scaffold group (first-N).
    """
    if max_samples_per_scaffold is None:
        return np.arange(scaffolds.shape[0])
    keep = limit_scaffold_representation(scaffolds, max_samples_per_scaffold)
    return keep


def joint_scaffold_kfold_split(
    activity_ds,
    pose_ds,
    k: int,
    inner_val_frac: float = 0.5,
    max_samples_per_scaffold: Optional[int] = None,
    seed: int = 0,
) -> List[tuple[Split, Split]]:
    """
    Produce synchronized K-fold scaffold splits across BOTH datasets.
    No scaffold appearing in test/val appears in train (strict), and
    for each fold the scaffold sets for activity and pose match per split.

    Returns
    -------
    List of tuples: (activity_split, pose_split) for each fold.
    """
    # 1) Per-dataset scaffold arrays
    act_scaffolds_full = _scaffold_array(activity_ds)
    pose_scaffolds_full = _scaffold_array(pose_ds)

    # 2) Optional per-scaffold limiting (done independently per dataset)
    act_keep = _index_limit_by_scaffold(act_scaffolds_full, max_samples_per_scaffold)
    pose_keep = _index_limit_by_scaffold(pose_scaffolds_full, max_samples_per_scaffold)

    act_scaffolds = act_scaffolds_full[act_keep]
    pose_scaffolds = pose_scaffolds_full[pose_keep]

    # 3) Concatenate to run ONE GroupKFold by scaffold across both datasets
    groups_concat = np.concatenate([act_scaffolds, pose_scaffolds])
    n_act = act_scaffolds.shape[0]

    _X = np.zeros((groups_concat.shape[0], 1))
    gkf = GroupKFold(k)

    splits_joint: List[tuple[Split, Split]] = []

    for train_concat, test_concat in gkf.split(_X, groups=groups_concat):
        # Map concat indices back to per-dataset *local* indices
        act_train_local = train_concat[train_concat < n_act]
        act_test_local  = test_concat[test_concat  < n_act]

        pose_train_local = train_concat[train_concat >= n_act] - n_act
        pose_test_local  = test_concat[test_concat  >= n_act] - n_act

        # 4) Split each dataset's test into val/test
        act_val_local, act_test_local2   = _split_random(act_test_local,  inner_val_frac, seed=seed)
        pose_val_local, pose_test_local2 = _split_random(pose_test_local, inner_val_frac, seed=seed)

        # 5) Map local -> original dataset indices
        act_train_idx = act_keep[act_train_local]
        act_val_idx   = act_keep[act_val_local]
        act_test_idx  = act_keep[act_test_local2]

        pose_train_idx = pose_keep[pose_train_local]
        pose_val_idx   = pose_keep[pose_val_local]
        pose_test_idx  = pose_keep[pose_test_local2]

        split_act  = Split(act_train_idx,  act_val_idx,  act_test_idx)
        split_pose = Split(pose_train_idx, pose_val_idx, pose_test_idx)

        splits_joint.append((split_act, split_pose))

    # 6) Safety: assert no leakage and matched scaffold sets per split part
    def _scaffold_set(scaffolds, idx): return set(scaffolds[idx])
    _assert_no_scaffold_leakage(act_scaffolds_full, [Split(act_keep[s.train_split], act_keep[s.val_split], act_keep[s.test_split]) for s,_ in splits_joint], tag="joint-activity")
    _assert_no_scaffold_leakage(pose_scaffolds_full,[Split(pose_keep[s.train_split],pose_keep[s.val_split],pose_keep[s.test_split]) for _,s in splits_joint], tag="joint-pose")

    for i, (sa, sp) in enumerate(splits_joint):
        for name, a_idx, p_idx in [
            ("train", sa.train_split, sp.train_split),
            ("val",   sa.val_split,   sp.val_split),
            ("test",  sa.test_split,  sp.test_split),
        ]:
            aset = _scaffold_set(act_scaffolds_full, a_idx)
            pset = _scaffold_set(pose_scaffolds_full, p_idx)
            if aset != pset:
                # Non-fatal: this could only happen if a scaffold had zero members in one dataset
                # because of limiting. Typically shouldn't happen; raise to be explicit.
                raise RuntimeError(f"[joint] fold {i} {name}: activity/pose scaffold sets differ "
                                   f"(|A|={len(aset)}, |P|={len(pset)})")

    # Optional: print per-fold stats for sanity
    print_scaff = False
    if print_scaff:
        _print_scaffold_stats_per_fold(act_scaffolds_full, [s[0] for s in splits_joint], tag="joint-activity")
        _print_scaffold_stats_per_fold(pose_scaffolds_full, [s[1] for s in splits_joint], tag="joint-pose")

    return splits_joint


from collections import Counter

def report_scaffold_distribution(
    fold_id: int,
    dataset_name: str,
    split: Split,
    scaffolds: np.ndarray,
    tag: str = "scaffold",
    topn: int = 5,
):
    """
    Print scaffold distribution for each partition of one fold.
    Also checks that no scaffold leaks across train/val/test.

    Parameters
    ----------
    fold_id : int
        Fold index.
    dataset_name : str
        'activity' or 'pose'.
    split : Split
        Train/val/test indices for this dataset.
    scaffolds : np.ndarray
        Scaffold array aligned with dataset.
    tag : str
        Label for the group (e.g., 'scaffold', 'pocket').
    topn : int
        Number of top scaffolds to print.
    """
    def _summary(name, idx):
        arr = scaffolds[idx]
        total = len(arr)
        cnt = Counter(arr)
        lines = [f"[fold {fold_id} | {dataset_name:8s} | {name:5s}] total={total}"]
        for scaf, n in cnt.most_common(topn):
            share = 100.0 * n / total
            lines.append(f"  {tag:8s} {scaf:4s}: n={n:6d}  share={share:5.2f}%")
        return "\n".join(lines)

    print(_summary("train", split.train_split))
    print(_summary("val",   split.val_split))
    print(_summary("test",  split.test_split))

    # --- leakage check ---
    train_set = set(scaffolds[split.train_split])
    val_set   = set(scaffolds[split.val_split])
    test_set  = set(scaffolds[split.test_split])
    assert train_set.isdisjoint(val_set), f"Leakage train↔val in {dataset_name} fold {fold_id}"
    assert train_set.isdisjoint(test_set), f"Leakage train↔test in {dataset_name} fold {fold_id}"
    #assert val_set.isdisjoint(test_set),   f"Leakage val↔test in {dataset_name} fold {fold_id}"












    
