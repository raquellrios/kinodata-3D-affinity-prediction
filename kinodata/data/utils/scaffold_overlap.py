
# kinodata/data/utils/scaffold_overlap.py
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Set, Optional
import numpy as np

from ..data_split import Split  # your existing class

@dataclass
class OverlapStats:
    n_scaf_act: int
    n_scaf_pose: int
    n_union: int
    n_inter: int
    jaccard: float

def _scaffolds_list(ds) -> List[str]:
    # ds is a torch_geometric InMemoryDataset or your wrapper; items have .scaffold
    return [getattr(ds[i], "scaffold") for i in range(len(ds))]

def _subset_scaffolds(scaffolds: List[str], indices: Iterable[int]) -> List[str]:
    idx = np.array(list(indices), dtype=int)
    return [scaffolds[i] for i in idx]

def _stats(act_scaffolds: Iterable[str], pose_scaffolds: Iterable[str]) -> OverlapStats:
    A, P = set(act_scaffolds), set(pose_scaffolds)
    U = A | P
    I = A & P
    return OverlapStats(len(A), len(P), len(U), len(I), len(I) / max(1, len(U)))

def _top_only(set_a: Set[str], set_b: Set[str], k=15) -> List[Tuple[str, int]]:
    # return a few frequent scaffolds that are only in A not in B
    return list(zip(list(set_a - set_b)[:k], [0]*min(k, len(set_a - set_b))))

def summarize_overlap(
    activity_ds,
    pose_ds,
    split_act: Optional[Split] = None,
    split_pose: Optional[Split] = None,
) -> Dict[str, OverlapStats]:
    act_scaf_all  = _scaffolds_list(activity_ds)
    pose_scaf_all = _scaffolds_list(pose_ds)

    out: Dict[str, OverlapStats] = {}
    out["GLOBAL"] = _stats(act_scaf_all, pose_scaf_all)

    if split_act is None or split_pose is None:
        return out

    parts = [("TRAIN", split_act.train_split, split_pose.train_split),
             ("VAL",   split_act.val_split,   split_pose.val_split),
             ("TEST",  split_act.test_split,  split_pose.test_split)]

    for name, ia, ip in parts:
        act_sub  = _subset_scaffolds(act_scaf_all, ia)
        pose_sub = _subset_scaffolds(pose_scaf_all, ip)
        out[name] = _stats(act_sub, pose_sub)

    return out

def pretty_print(stats: Dict[str, OverlapStats]) -> None:
    for k, s in stats.items():
        print(f"[{k}]  act={s.n_scaf_act:5d}  pose={s.n_scaf_pose:5d}  "
              f"∪={s.n_union:5d}  ∩={s.n_inter:5d}  Jaccard={s.jaccard:0.3f}")