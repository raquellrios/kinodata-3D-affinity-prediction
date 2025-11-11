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

    #max_samps = getattr(config, "max_samples_per_scaffold", 1000)
    max_samps = getattr(config, "max_samples_per_scaffold", 3000)
    print(f"the max samples per scaffold are {max_samps}")
    combined_splits = unified_scaffold_splits(
    activity_ds, pose_ds,
    split_type=config.split_type,
    k=config.k_fold,
    max_samples_per_scaffold=max_samps
    )
    fold_split = combined_splits[config.split_index]
    split_act = fold_split['activity']
    split_pose = fold_split['pose']

    for part in ("train", "val", "test"):
        # get the index lists
        idx_act  = getattr(split_act,  f"{part}_split")
        idx_pose = getattr(split_pose, f"{part}_split")

        # collect the raw scaffold strings
        scaff_act  = [activity_ds[i].scaffold for i in idx_act]
        scaff_pose = [pose_ds[i].scaffold     for i in idx_pose]

        set_act  = set(scaff_act)
        set_pose = set(scaff_pose)

        print(f"\n=== {part.upper()} ===")
        print(f"  Activity: {len(idx_act)} mols, {len(set_act)} unique scaffolds")
        print(f"  Pose:     {len(idx_pose)} mols, {len(set_pose)} unique scaffolds")
        print(f"  Scaffolds match? {set_act == set_pose}")

        match = set_act == set_pose
        print(f"  Scaffolds match? {match}")
        if not match:
            only_in_act = sorted(set_act  - set_pose)
            only_in_pose= sorted(set_pose - set_act)
            print(f"    → In activity only ({len(only_in_act)}): {only_in_act[:10]}{'...' if len(only_in_act)>10 else ''}")
            print(f"    → In pose only     ({len(only_in_pose)}): {only_in_pose[:10]}{'...' if len(only_in_pose)>10 else ''}")

    # Build scaffold→fold maps for each dataset
    def get_fold_map(split, ds):
        # invert the .scaffold labels to a dict: scaffold → fold name
        fmap = {}
        for part in ("train", "val", "test"):
            idxs = getattr(split, f"{part}_split")
            for i in idxs:
                fmap[ds[i].scaffold] = part
        return fmap

    map_act  = get_fold_map(split_act,  activity_ds)
    map_pose = get_fold_map(split_pose, pose_ds)

    # Find only the common scaffolds
    common = set(map_act).intersection(map_pose)

    mismatches = [s for s in common if map_act[s] != map_pose[s]]

    print(f"Common scaffolds: {len(common)}")
    print(f"Mismatches:      {len(mismatches)}")
    if mismatches:
        print("Examples of inconsistent assignments:", mismatches[:10])
    else:
        print("All shared scaffolds are in the same fold!")

    print("\n--- Detailed mismatch report ---")
    for smi in mismatches:
        # Find all activity indices with this scaffold
        act_idxs = [i for i, d in enumerate(activity_ds) if d.scaffold == smi]
        # Find all pose indices with this scaffold
        pose_idxs = [i for i, d in enumerate(pose_ds)     if d.scaffold == smi]

        # Lookup fold names
        fold_act  = map_act[smi]
        fold_pose = map_pose[smi]

        print(f"\nScaffold: {smi!r}")
        print(f"  Activity: indices {act_idxs}, assigned to '{fold_act}'")
        print(f"  Pose:     indices {pose_idxs}, assigned to '{fold_pose}'")


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
