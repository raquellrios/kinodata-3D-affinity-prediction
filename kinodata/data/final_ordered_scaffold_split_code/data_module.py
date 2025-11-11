from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.lightning_datamodule import LightningDataset
#from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.transforms import Compose

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.grouped_split import KinodataKFoldSplit
from sklearn.preprocessing import StandardScaler
from kinodata.data.grouped_split import unified_scaffold_splits
#from kinodata.data.grouped_split import print_scaffolds_in_splits, save_scaffolds_to_csv, count_scaffold_distribution, visualise_scaffold_overlap

from kinodata.data.dataset import (
    KinodataDocked,
    Filtered,
)
from kinodata.data.dataset_davids_data import (
    DavidsdataDocked,
    Filtered_david,
)
import kinodata.transform as T
from kinodata.types import NodeType


import torch
from torch.utils.data import Dataset, DataLoader #, CombinedLoader
#from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
#from pytorch_lightning.trainer.supporters import CombinedLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl

from rdkit import Chem
from rdkit.Chem import Descriptors
import os


Kwargs = Dict[str, Any]

import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold
from typing import Tuple

def make_joint_scaffold_splits(
    activity_ds,
    pose_ds,
    k: int,
    split_index: int = 0,
    val_frac: float = 0.5,
    seed: int = 0,
    max_samples_per_scaffold_activity: int | None = None,
    max_samples_per_scaffold_pose: int | None = None,
):
    """
    Build ONE scaffold-based K-fold partition over the UNION of both datasets,
    then derive per-dataset splits by selecting items whose scaffold belongs to
    the train/val/test group sets for the chosen fold.

    Returns
    -------
    split_act, split_pose : Split
        Two Split objects (one for activity, one for pose) that are aligned.
    """

    # --- 1) Collect scaffolds and "local indices" for each dataset ---
    scaff_act = np.array([activity_ds[i].scaffold for i in range(len(activity_ds))], dtype=object)
    scaff_pose = np.array([pose_ds[i].scaffold     for i in range(len(pose_ds))],     dtype=object)

    idx_act   = np.arange(len(activity_ds))
    idx_pose  = np.arange(len(pose_ds))

    # --- 2) (Optional) cap samples per scaffold per dataset to de-overrepresent huge groups ---
    def cap_per_scaffold(scaff, idx, cap):
        if cap is None:
            return idx
        keep = []
        # group by scaffold
        from collections import defaultdict
        bucket = defaultdict(list)
        for i, s in zip(idx, scaff[idx]):
            bucket[s].append(i)
        for s, ids in bucket.items():
            keep.extend(ids[:cap])
        return np.array(keep, dtype=int)

    keep_act  = cap_per_scaffold(scaff_act, idx_act,  max_samples_per_scaffold_activity)
    keep_pose = cap_per_scaffold(scaff_pose, idx_pose, max_samples_per_scaffold_pose)

    # Use possibly-reduced sets for building the joint fold
    scaff_act_kept  = scaff_act[keep_act]
    scaff_pose_kept = scaff_pose[keep_pose]

    # --- 3) Build the joint table used for folding ---
    # We concatenate samples from both datasets; GroupKFold will respect the scaffold groups.
    joint_scaffolds = np.concatenate([scaff_act_kept, scaff_pose_kept])
    # We also keep track of the origin and original indices so we can map back:
    src_flags = np.concatenate([
        np.zeros_like(scaff_act_kept, dtype=np.int8),  # 0 = activity
        np.ones_like(scaff_pose_kept, dtype=np.int8),  # 1 = pose
    ])
    src_local_idx = np.concatenate([keep_act, keep_pose])

    # --- 4) Outer scaffold K-fold on the joint samples ---
    gkf = GroupKFold(n_splits=k)
    X_dummy = np.zeros((len(joint_scaffolds), 1))
    folds = list(gkf.split(X_dummy, groups=joint_scaffolds))
    assert 0 <= split_index < k, f"split_index must be in [0,{k-1}]"

    train_mask, heldout_mask = folds[split_index]

    # Unique scaffold sets for the chosen outer fold:
    heldout_scaffolds = np.unique(joint_scaffolds[heldout_mask])
    train_scaffolds   = np.unique(joint_scaffolds[train_mask])

    # --- 5) Split the heldout scaffolds into val/test (by scaffold) ---
    rng = default_rng(seed)
    perm = rng.permutation(len(heldout_scaffolds))
    pivot = int(len(heldout_scaffolds) * val_frac)
    val_scaffolds  = set(heldout_scaffolds[perm[:pivot]])
    test_scaffolds = set(heldout_scaffolds[perm[pivot:]])
    train_scaffolds = set(train_scaffolds)

    # --- 6) Map scaffold sets back to per-dataset indices ---
    def select_indices(scaff_all, idx_all, which: set[str]):
        mask = np.array([s in which for s in scaff_all[idx_all]], dtype=bool)
        return idx_all[mask]

    # Activity indices per split:
    act_train_idx = select_indices(scaff_act, idx_act,  train_scaffolds)
    act_val_idx   = select_indices(scaff_act, idx_act,  val_scaffolds)
    act_test_idx  = select_indices(scaff_act, idx_act,  test_scaffolds)

    # Pose indices per split:
    pose_train_idx = select_indices(scaff_pose, idx_pose, train_scaffolds)
    pose_val_idx   = select_indices(scaff_pose, idx_pose, val_scaffolds)
    pose_test_idx  = select_indices(scaff_pose, idx_pose, test_scaffolds)

    # --- 7) Build Split objects ---
    split_act  = Split(act_train_idx,  act_val_idx,  act_test_idx)
    split_pose = Split(pose_train_idx, pose_val_idx, pose_test_idx)

    # --- 8) (Optional) quick summary + overlap checks ---
    def summarize(tag, ds_scaff, tr, va, te):
        def nuniq(ix): return len(set(ds_scaff[i] for i in ix))
        def largest(ix):
            from collections import Counter
            c = Counter(ds_scaff[i] for i in ix)
            return c.most_common(1)[0][1] if c else 0
        print(f"[joint-scaffold] {tag}")
        print(f"  train: {nuniq(tr):4d} scaffolds, {len(tr):5d} mols (largest={largest(tr)})")
        print(f"  val  : {nuniq(va):4d} scaffolds, {len(va):5d} mols (largest={largest(va)})")
        print(f"  test : {nuniq(te):4d} scaffolds, {len(te):5d} mols (largest={largest(te)})")

    summarize("activity",
              scaff_act, act_train_idx, act_val_idx, act_test_idx)
    summarize("pose",
              scaff_pose, pose_train_idx, pose_val_idx, pose_test_idx)

    # Overlap of pose vs activity (per split), relative to pose:
    def overlap_ratio(pose_ix, act_ix, scaff_pose_all, scaff_act_all, name):
        pose_sc = set(scaff_pose_all[i] for i in pose_ix)
        act_sc  = set(scaff_act_all[i] for i in act_ix)
        inter = pose_sc & act_sc
        den = len(pose_sc) if pose_sc else 1
        print(f"  [{name}] pose scaffolds: {len(pose_sc)}, overlap with activity: {len(inter)} ({len(inter)/den:.1%})")

    print("[joint-scaffold] pose→activity scaffold coverage:")
    overlap_ratio(pose_train_idx, act_train_idx, scaff_pose, scaff_act, "train")
    overlap_ratio(pose_val_idx,   act_val_idx,   scaff_pose, scaff_act, "val")
    overlap_ratio(pose_test_idx,  act_test_idx,  scaff_pose, scaff_act, "test")

    return split_act, split_pose

def _scaffold_sets(scaffolds: np.ndarray, idx_train, idx_val, idx_test):
    S_tr = set(scaffolds[idx_train]) if idx_train is not None else set()
    S_va = set(scaffolds[idx_val])   if idx_val   is not None else set()
    S_te = set(scaffolds[idx_test])  if idx_test  is not None else set()
    return S_tr, S_va, S_te


def assert_no_leakage(scaffolds: np.ndarray, split, tag: str):
    """Raise if any scaffold appears both in train and val/test."""
    S_tr, S_va, S_te = _scaffold_sets(scaffolds, split.train_split, split.val_split, split.test_split)
    leak_tv = S_tr & S_va
    leak_tt = S_tr & S_te
    if leak_tv or leak_tt:
        raise RuntimeError(
            f"[leakage:{tag}] found scaffolds in train ∩ val: {len(leak_tv)} "
            f"and train ∩ test: {len(leak_tt)}"
        )
    # Optional: quick summary
    print(f"[no-leak:{tag}] |train|={len(S_tr)} |val|={len(S_va)} |test|={len(S_te)} "
          f"overlap(train,val)={len(leak_tv)} overlap(train,test)={len(leak_tt)}")

def _check_pose_vs_activity_scaffolds(split_act, split_pose, activity_ds, pose_ds):
    def get_scaffolds(dataset, indices):
        return set(dataset[i].scaffold for i in indices)

    for split_name, act_idx, pose_idx in [
        ("train", split_act.train_split, split_pose.train_split),
        ("val",   split_act.val_split,   split_pose.val_split),
        ("test",  split_act.test_split,  split_pose.test_split),
    ]:
        act_scaffolds  = get_scaffolds(activity_ds, act_idx)
        pose_scaffolds = get_scaffolds(pose_ds, pose_idx)

        overlap = pose_scaffolds & act_scaffolds
        ratio = len(overlap) / len(pose_scaffolds) if pose_scaffolds else 0.0

        print(f"[{split_name}] pose scaffolds: {len(pose_scaffolds)}")
        print(f"           overlap with activity: {len(overlap)} "
              f"({ratio:.1%})")
        if overlap:
            print(f"           examples: {list(overlap)[:5]}")

def save_scaffold_molwt(
    dataset,
    split: Split,
    #out_csv: str,
    #smiles_attr: str = "smiles",
    #scaffold_attr: str = "scaffold",
):
    """
    Build a DataFrame of {split, scaffold, mol_weight} and save to CSV.
    
    Args:
        dataset: your InMemoryDataset (activity_ds or pose_ds).
        split:    a Split object with train/val/test index lists.
        out_csv:  path to write the resulting CSV.
        smiles_attr:    name of the SMILES field on each data object.
        scaffold_attr:  name of the scaffold field on each data object.
    """
    #records = []
    #for phase in ("train", "val", "test"):
        #idxs = getattr(split, f"{phase}_split")
        #for i in idxs:
            #data = dataset[i]
            #print(data)


            #smi = getattr(data, smiles_attr)
            #scf = getattr(data, scaffold_attr)
            #mol = Chem.MolFromSmiles(smi)
            #mw = Descriptors.MolWt(mol) if mol is not None else np.nan
            #records.append({
            #    "split":     phase,
            #    "scaffold":  scf,
            #    "mol_weight": mw,
            #})

    #df = pd.DataFrame(records)
    # ensure directory exists
    #os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    #df.to_csv(out_csv, index=False)
    #print(f"Saved scaffold–molwt table to {out_csv}")







def assert_unique_value(key: str, *kwarg_dicts: Optional[Kwargs], msg: str = ""):
    values = []
    for kwarg_dict in kwarg_dicts:
        if not kwarg_dict:
            continue
        if key in kwarg_dict:
            values.append(kwarg_dict[key])
    assert len(set(values)) <= 1, msg



def make_data_module(
    split: Split,
    batch_size: int,
    num_workers: int,
    dataset_cls: Optional[type[InMemoryDataset]] = None,
    dataset_instance: Optional[InMemoryDataset] = None,
    train_kwargs: Kwargs | None = None,
    val_kwargs: Optional[Kwargs] = None,
    test_kwargs: Optional[Kwargs] = None,
    one_time_transform: Optional[Callable[[InMemoryDataset], InMemoryDataset]] = None,
    normalization: bool = False,
    **kwargs,
    ) -> LightningDataset:

    assert (dataset_cls is None) ^ (dataset_instance is None)
    
    def subset(full_ds: InMemoryDataset, idx: List[int]):
        view = full_ds[idx]
        return one_time_transform(view) if one_time_transform else view

    if dataset_instance is not None:
        full_dataset=dataset_instance
    else:
        full_dataset = dataset_cls(**(train_kwargs or {}))

    train_dataset = subset(full_dataset, split.train_split)
    val_dataset = subset(full_dataset, split.val_split) if split.val_split else None
    test_dataset = subset(full_dataset, split.test_split) if split.test_split else None
    
    # --- after train_dataset/val_dataset/test_dataset are created ---
    train_dataset.transform = (train_kwargs or {}).get("transform")
    val_dataset.transform   = (val_kwargs   or {}).get("transform")
    test_dataset.transform  = (test_kwargs  or {}).get("transform")

    if normalization and train_dataset is not None:
        
        y = torch.stack([d.y for d in train_dataset])
        mu, sigma = y.mean().item(), y.std().item()

        print("verify normalization of activities")
        print("mean "+str(y.mean().item())+" std "+str(y.std().item()))
        

        class ScaleY:
            def __init__(self, mu, sigma):
                self.mu = mu
                self.sigma = sigma
            def __call__(self, data):
                data.y = (data.y - mu) / sigma
                return data
            
        norm_tf = ScaleY(mu, sigma)

        #apply once --  no cloning

        def _add(tf_new, tf_old):
            # chain new→old so scaling happens before any other transform
            return Compose([tf_new, tf_old]) if tf_old is not None else tf_new

        train_dataset.transform = _add(norm_tf, train_dataset.transform)
        val_dataset.transform   = _add(norm_tf, val_dataset.transform)
        test_dataset.transform  = _add(norm_tf, test_dataset.transform)

        normalized_values = torch.tensor([d.y.item() for d in train_dataset])
        print("**After Normalization**")
        print(f"  Mean: {normalized_values.mean().item():.4f}, Std: {normalized_values.std().item():.4f}")
        print(f"  Sample values: {normalized_values[:5].squeeze().tolist()}")  # Print first few values
        

    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True, 
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
        **kwargs,
    )


def compose(transforms: Optional[list]) -> Optional[Callable]:
    return None if transforms is None else Compose(transforms)


def load_precomputed_split(config) -> Split:
    if not Path(config.data_split).exists():
        raise FileNotFoundError(config.data_split)
    print(f"Loading split from {config.data_split}..")
    # split that assigns *idents* to train/val/test
    split = Split.from_data_frame(pd.read_csv(config.data_split))
    split.source_file = str(config.data_split)
    return split



class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, dm_a, dm_b):
        super().__init__()
        self.dm_a = dm_a  # Data module for activity dataset
        self.dm_b = dm_b  # Data module for pose dataset
        

    def setup(self, stage=None):
        self.dm_a.setup(stage=stage)
        self.dm_b.setup(stage=stage)


    print(CombinedLoader.__init__.__doc__)

    def train_dataloader(self):
         return CombinedLoader(
             {
                 "activity": self.dm_a.train_dataloader(),
                 "pose": self.dm_b.train_dataloader(),
             },
             mode='max_size',  
         )
    
    #do i WANT HERE A shuffle=True ? WHERE TO PUT IT THEN?

    def val_dataloader(self):
         return CombinedLoader(
             {
                 "activity": self.dm_a.val_dataloader(),
                 "pose": self.dm_b.val_dataloader(),
             },
             mode='sequential',  
         )

    def test_dataloader(self):
         return CombinedLoader(
             {
                 "activity": self.dm_a.test_dataloader(),
                 "pose": self.dm_b.test_dataloader(),
             },
             mode='sequential',
         )





def make_kinodata_module(
    config: Config, transforms=None, one_time_transform=None
        ) -> LightningDataset:
    
    
    ###note that the K-FOLD splitting is happening before merging the two datasets!
    poseBaseCls= partial(DavidsdataDocked, remove_hydrogen=config.remove_hydrogen)
    
    activityBaseCls = partial(KinodataDocked, remove_hydrogen=config.remove_hydrogen)
    
    
    if config.filter_rmsd_max_value is not None: #not sure if we want this anymore, since now we may not want to filter by RMSD? only appliying it to the kinodata dataset 
        poseCls = Filtered_david(
            poseBaseCls(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )
        activityCls = Filtered(
            activityBaseCls(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )

    else:
        poseCls, activityCls = poseBaseCls, activityBaseCls

    pose_ds = poseCls()
    activity_ds = activityCls()
  
    #print(type(config))
    #print("above is the type of config")
    #
    #config["perturb_ligand_positions"] = 0.01
    #config["perturb_complex_positions"] = 0.5
    

    #print(f"Perturb ligand positions: {config.perturb_ligand_positions}")
    #print(f"Postions reaching the make_kinodata_module condition positions: {config.need_distances}")



    if transforms is None:
        transforms = []

    augmentations = []

    if config.perturb_ligand_positions and config.need_distances:
        augmentations.append(
            T.PerturbAtomPositions(NodeType.Ligand, std=config.perturb_ligand_positions)
        )
        print(f"Ligand perturbation transformation added with std={config.perturb_ligand_positions}")
    if config.perturb_pocket_positions and config.need_distances:
        augmentations.append(
            T.PerturbAtomPositions(NodeType.Pocket, std=config.perturb_pocket_positions)
        )
    if "perturb_complex_positions" in config and config.perturb_complex_positions > 0.0:
        augmentations.append(
            T.PerturbAtomPositions(
                NodeType.Complex, std=config.perturb_complex_positions
            )
        )
        print(f"Ligand complex positions transformation added with std={config.perturb_complex_positions}")

    if config.need_distances:
        ...

    if config.add_docking_scores:
        assert config.need_distances
        raise NotImplementedError

    train_transform = compose(augmentations + transforms)
    print(f"Training transforms: {train_transform}")
    val_transform = compose(transforms)

    def get_split(raw):
        if config.data_split:
            sp = load_precomputed_split(config)
            return sp.remap_index(raw.ident_index_map())
        else:
            splits = KinodataKFoldSplit(config.split_type, config.k_fold).split(raw)
            return splits[config.split_index]

    #split_act = get_split(activity_ds)
    #split_pose = get_split(pose_ds)

    split_act, split_pose = make_joint_scaffold_splits(
    activity_ds,
    pose_ds,
    k=config.k_fold,
    split_index=config.split_index,
    val_frac=0.5,                  # keep your current 50/50 val/test split
    seed=0,
    max_samples_per_scaffold_activity=None,   # or e.g. 3000 if you want a cap on activity too
    max_samples_per_scaffold_pose=3000,       # your preferred cap for pose
)
    
    scaff_act = np.array([activity_ds[i].scaffold for i in range(len(activity_ds))], dtype=object)
    scaff_pose = np.array([pose_ds[i].scaffold     for i in range(len(pose_ds))],     dtype=object)

    # 1) Leakage checks (per dataset, this fold)
    assert_no_leakage(scaff_act, split_act,  tag="activity")
    assert_no_leakage(scaff_pose, split_pose, tag="pose")

    #max_samps = getattr(config, "max_samples_per_scaffold", 1000)
    #max_samps = getattr(config, "max_samples_per_scaffold", None)
    
    print(f"Split kinodata: Train size {split_act.train_size}, Val size {split_act.val_size}, Test size {split_act.test_size}")
    print(f"Split kinodocked: Train size {split_pose.train_size}, Val size {split_pose.val_size}, Test size {split_pose.test_size}")

    _check_pose_vs_activity_scaffolds(split_act, split_pose, activity_ds, pose_ds)

    save_scaffold_molwt(activity_ds, split_act)

    num_workers_config = getattr(config, 'num_workers', 1)
    print(f"the number of workers selected for both datasets are {num_workers_config}")

    data_module_1 = make_data_module(
        split_act,
        config.batch_size,
        num_workers = num_workers_config,
        dataset_instance=activity_ds,
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform}, #is this okay?
        one_time_transform=one_time_transform,
        normalization=True
    )

    data_module_2 = make_data_module(
        split_pose,
        config.batch_size, 
        num_workers = num_workers_config,
        dataset_instance=pose_ds, 
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform}, #is this okay?
        one_time_transform=one_time_transform,
    )

    # Combine both data modules
    combined_data_module = CombinedDataModule(data_module_1, data_module_2)

    return combined_data_module
