from pathlib import Path
from functools import singledispatchmethod
from typing import List, Optional, Iterable, Tuple, Protocol
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


# assumes:
# Split = namedtuple("Split", ["train_index", "val_index", "test_index"])
# _split_random(test_index: np.ndarray, frac_val: float) -> Tuple[np.ndarray, np.ndarray]
# limit_scaffold_representation(groups: np.ndarray, max_per: int) -> Union[np.ndarray, np.bool_]

def _as_indices(mask_or_idx: np.ndarray, n: int) -> np.ndarray:
    """Normalize boolean mask or integer indices to integer indices."""
    mask_or_idx = np.asarray(mask_or_idx)
    if mask_or_idx.dtype == bool:
        if mask_or_idx.shape[0] != n:
            raise ValueError("Boolean mask length does not match array length.")
        return np.nonzero(mask_or_idx)[0]
    return mask_or_idx.astype(np.int64, copy=False)

def _generator_to_list(generator: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List["Split"]:
    # unchanged behavior: uses given indices as-is
    return [Split(train_index, *_split_random(test_index, 0.5))  # type: ignore
            for train_index, test_index in generator]

def group_k_fold_split(
    group_index: np.ndarray,
    k: int,
    max_samples_per_scaffold: Optional[int] = None,
) -> List["Split"]:
    """
    GroupKFold splits with optional downsampling of overrepresented scaffolds.
    Returns global indices so they map to the original dataset.
    """
    group_index = np.asarray(group_index).ravel()
    n = group_index.shape[0]
    global_idx = np.arange(n)

    # Optional filtering: compute a view into the original dataset
    if max_samples_per_scaffold is not None:
        keep = limit_scaffold_representation(group_index, max_samples_per_scaffold)
        keep = _as_indices(np.asarray(keep), n)
        group_index_f = group_index[keep]
        global_idx = global_idx[keep]
    else:
        group_index_f = group_index  # no filtering

    # Safety: GroupKFold requires at least k unique groups
    n_unique = np.unique(group_index_f).size
    if k > n_unique:
        raise ValueError(f"k={k} is larger than number of unique groups ({n_unique}).")

    # Fit GroupKFold on the (possibly) filtered view
    gkf = GroupKFold(n_splits=k)
    _X = np.zeros((group_index_f.shape[0], 1))
    gen = gkf.split(_X, groups=group_index_f)

    # Map fold indices back to original/global indices before producing Split objects
    mapped_gen = ((global_idx[tr], global_idx[te]) for tr, te in gen)
    return _generator_to_list(mapped_gen)




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
    


from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class DualSplit:
    activity: Split
    pose: Split

def _cluster_pockets_union(
    activity_ds,
    pose_ds,
    clustering: AffinityPropagation,
    sim_measure: BLOSUMSubstitutionSimilarity,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return per-example pocket cluster ids for activity and pose datasets,
    computed from a single clustering fit on the UNION of pocket sequences.
    """
    # 1) Collect sequences
    act_seq = np.asarray(activity_ds.data.pocket_sequence)
    pose_seq = np.asarray(pose_ds.data.pocket_sequence)

    union_df = pd.DataFrame(
        {"index": np.arange(len(act_seq) + len(pose_seq)),
         "pocket_sequence": np.concatenate([act_seq, pose_seq], axis=0)}
    )

    # 2) Cluster on union
    df_labels = clustering(
        union_df,
        "pocket_sequence",
        fn_similarity=sim_measure(),
    ).sort_values(by="index", ascending=True)

    print("df labels are")
    print(df_labels) #what is df labels?

    labels = np.asarray(df_labels[clustering.cluster_key].values)
    act_labels = labels[:len(act_seq)]
    pose_labels = labels[len(act_seq):]
    return act_labels, pose_labels

def pocket_overlap_report(activity_groups: np.ndarray, pose_groups: np.ndarray, tag="GLOBAL"):
    A, P = set(activity_groups.tolist()), set(pose_groups.tolist())
    U, I = A | P, A & P
    j = (len(I) / len(U)) if U else 0.0
    print(f"[{tag}]  act={len(A):5d}  pose={len(P):5d}  ∪={len(U):5d}  ∩={len(I):5d}  Jaccard={j:.3f}")


def joint_pocket_kfold_split(
    activity_ds,
    pose_ds,
    k: int,
    inner_val_frac: float = 0.5,
    max_samples_per_pocket: Optional[int] = None,
    seed: int = 0,
    clustering: AffinityPropagation = KinodataKFoldSplit.pocket_clustering,
    sim_measure_cls=BLOSUMSubstitutionSimilarity,
) -> List[DualSplit]:
    """
    Produce synchronized K-fold pocket splits across BOTH datasets.
    No pocket in test appears in train/val in either task (strict eval).
    """
    # 1) Pocket ids for each example, consistent across datasets
    act_groups, pose_groups = _cluster_pockets_union(
        activity_ds, pose_ds, clustering, sim_measure_cls
    )
    print("act groups is")
    print(act_groups)

    print("pose groups is")
    print(pose_groups)

    #pocket_overlap_report(act_groups, pose_groups, tag="GLOBAL")

    # 2) (Optional) downsample overrepresented pockets on the concatenated view
    #def _limit(groups: np.ndarray, max_per: int) -> np.ndarray:
        # Generic version of limit_scaffold_representation
    #    df = pd.DataFrame({"g": groups}).value_counts()
    #    keep_idx = []
    #    for g, cnt in df.items():
    #        idx = np.where(groups == g)[0]
    #        keep_idx.extend(idx[:max_per])
    #    return np.asarray(keep_idx, dtype=np.int64)

    # Build concatenated view
    nA, nP = len(act_groups), len(pose_groups)
    groups_all = np.concatenate([act_groups, pose_groups], axis=0)
    origin = np.concatenate([np.zeros(nA, dtype=np.int8), np.ones(nP, dtype=np.int8)], axis=0)
    local_idx = np.concatenate([np.arange(nA), np.arange(nP)], axis=0)

    # 2b) Downsample if requested
    #if max_samples_per_pocket is not None:
    #    keep = _limit(groups_all, max_samples_per_pocket)
    #    groups_all = groups_all[keep]
    #    origin     = origin[keep]
    #    local_idx  = local_idx[keep]

    # 3) GroupKFold on concatenated indices by pocket
    uniq = np.unique(groups_all).size

    print("len of unique pockets across al datasets")
    print(uniq)

    if k > uniq:
        raise ValueError(f"k={k} > unique pockets={uniq}")

    gkf = GroupKFold(n_splits=k)
    X = np.zeros((groups_all.shape[0], 1))
    folds: List[DualSplit] = []

    for tr_all, te_all in gkf.split(X, groups=groups_all):
        # Split test → val/test (consistent with your _generator_to_list behavior)
        # Keep deterministic split for reproducibility
        rng = default_rng(seed)
        te_perm = rng.permutation(te_all)
        pivot = int(len(te_perm) * inner_val_frac)
        va_all, ts_all = te_perm[:pivot], te_perm[pivot:]

        # Map back to per-dataset integer indices
        def sel(orig_flag: int, arr: np.ndarray) -> np.ndarray:
            m = origin[arr] == orig_flag
            return local_idx[arr][m]

        act_tr = sel(0, tr_all)
        act_va = sel(0, va_all)
        act_ts = sel(0, ts_all)

        pose_tr = sel(1, tr_all)
        pose_va = sel(1, va_all)
        pose_ts = sel(1, ts_all)

        # Build Split objects
        split_act  = Split(act_tr,  act_va,  act_ts)
        split_pose = Split(pose_tr, pose_va, pose_ts)
        folds.append(DualSplit(activity=split_act, pose=split_pose))

    # 4) Safety check: no pocket leakage across ANY train/val/test in either task
    def _assert_no_leak(groups: np.ndarray, sp: Split, tag: str):
        def pockets(idx): return set(groups[idx]) if idx is not None and len(idx) else set()
        T, V, S = pockets(sp.train_split), pockets(sp.val_split), pockets(sp.test_split)
        assert not (T & V), f"{tag}: train∩val"
        assert not (T & S), f"{tag}: train∩test"
        #assert not (V & S), f"{tag}: val∩test"

    for i, dual in enumerate(folds):
        _assert_no_leak(act_groups,  dual.activity, f"activity fold {i}")
        _assert_no_leak(pose_groups, dual.pose,     f"pose fold {i}")

    return folds, act_groups, pose_groups

def uniq_pockets(groups, idx):
    return set(groups[idx]) if idx is not None and len(idx) else set()


# groups: np.ndarray of pocket IDs per sample (activity or pose)
# idx: indices (np.ndarray) for a split: train/val/test
def _counts_per_pocket(groups: np.ndarray, idx: np.ndarray) -> pd.Series:
    if idx is None or len(idx) == 0:
        return pd.Series(dtype=int)
    return pd.Series(groups[idx]).value_counts().sort_values(ascending=False)

def _gini_from_counts(counts: np.ndarray) -> float:
    # Gini in [0,1]; 0 = perfectly uniform, 1 = totally dominated
    if counts.size == 0:
        return 0.0
    x = np.sort(counts)
    n = x.size
    cum = np.cumsum(x, dtype=float)
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(g)

def _entropy_from_counts(counts: np.ndarray) -> float:
    # Shannon entropy in nats; higher = more uniform
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def _split_name_parts() -> List[Tuple[str, str]]:
    return [("train", "train_split"), ("val", "val_split"), ("test", "test_split")]

# Build one tidy table for a single dataset (activity OR pose) for a given fold
def pocket_count_table_for_split(groups: np.ndarray, split, fold_idx: int, dataset_tag: str) -> pd.DataFrame:
    rows = []
    for split_name, attr in _split_name_parts():
        idx = getattr(split, attr)
        vc = _counts_per_pocket(groups, idx)
        total = int(vc.sum()) if not vc.empty else 0
        # dominance metrics
        gini = _gini_from_counts(vc.values) if total > 0 else 0.0
        ent  = _entropy_from_counts(vc.values) if total > 0 else 0.0
        for pocket_id, n in vc.items():
            share = n / total if total > 0 else 0.0
            rows.append({
                "fold": fold_idx,
                "dataset": dataset_tag,           # "activity" or "pose"
                "split": split_name,              # train/val/test
                "pocket_id": int(pocket_id),
                "n_mols": int(n),
                "share_in_split": float(share),
                "total_mols_in_split": total,
                "gini_split": gini,
                "entropy_split": ent,
            })
    return pd.DataFrame(rows)

# Build table across ALL folds for BOTH datasets (DualSplit list)
def pocket_count_table_all_folds(
    act_groups: np.ndarray,
    pose_groups: np.ndarray,
    dual_folds: List,
) -> pd.DataFrame:
    all_rows = []
    for i, dual in enumerate(dual_folds):
        df_a = pocket_count_table_for_split(act_groups,  dual.activity, i, "activity")
        df_p = pocket_count_table_for_split(pose_groups, dual.pose,     i, "pose")
        all_rows.append(df_a)
        all_rows.append(df_p)
    if not all_rows:
        return pd.DataFrame(columns=["fold","dataset","split","pocket_id","n_mols","share_in_split","total_mols_in_split","gini_split","entropy_split"])
    return pd.concat(all_rows, ignore_index=True)

# Convenience: print top-k dominating pockets per split per fold
def print_top_pockets(summary_df: pd.DataFrame, top_k: int = 5):
    if summary_df.empty:
        print("No data.")
        return
    for (fold, dataset, split), sub in summary_df.groupby(["fold","dataset","split"]):
        sub_sorted = sub.sort_values("n_mols", ascending=False).head(top_k)
        total = sub["total_mols_in_split"].max() if not sub.empty else 0
        gini  = sub["gini_split"].max() if not sub.empty else 0.0
        ent   = sub["entropy_split"].max() if not sub.empty else 0.0
        print(f"\n[fold {fold} | {dataset:8s} | {split:5s}] total={total}  gini={gini:.3f}  entropy={ent:.3f}")
        for _, r in sub_sorted.iterrows():
            print(f"  pocket {int(r.pocket_id):5d}: n={int(r.n_mols):6d}  share={r.share_in_split:6.2%}")



def _cap_indices_by_pocket(idx: np.ndarray,
                           groups: np.ndarray,
                           per_pocket_cap: int,
                           seed: int = 0) -> np.ndarray:
    """
    Cap number of samples per pocket (group) *within this index set*.
    idx: integer indices belonging to one split (train/val/test).
    groups: pocket id per sample for the full dataset.
    Returns a NEW array of capped indices (shuffled).
    """
    rng = default_rng(seed)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx

    # bucket indices by pocket (within this split only)
    pockets = {}
    for i in idx:
        pid = groups[i]
        pockets.setdefault(pid, []).append(i)

    kept = []
    for pid, bucket in pockets.items():
        b = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(b)
        kept.extend(b[:per_pocket_cap])

    kept = np.asarray(kept, dtype=np.int64)
    rng.shuffle(kept)  # optional global shuffle
    return kept


def cap_split_by_max_share(split: "Split",
                           groups: np.ndarray,
                           max_share: float,
                           min_cap: int = 1,
                           max_cap: Optional[int] = None,
                           seed: int = 0) -> "Split":
    """
    For each of train/val/test in `split`, compute a per-pocket cap = max_share * |split|.
    Apply that cap so no pocket exceeds this count *within that split*.
    Returns a NEW Split (does not modify input).
    """
    def cap_for(idx):
        n = len(idx) if idx is not None else 0
        base = max(int(np.floor(max_share * n)), min_cap)
        if max_cap is not None:
            base = min(base, max_cap)
        return base

    tr_cap = cap_for(split.train_split)
    va_cap = cap_for(split.val_split)
    te_cap = cap_for(split.test_split)

    tr_new = _cap_indices_by_pocket(split.train_split, groups, tr_cap, seed) if split.train_split is not None else None
    va_new = _cap_indices_by_pocket(split.val_split,   groups, va_cap, seed) if split.val_split   is not None else None
    te_new = _cap_indices_by_pocket(split.test_split,  groups, te_cap, seed) if split.test_split  is not None else None

    return Split(tr_new, va_new, te_new)














    
