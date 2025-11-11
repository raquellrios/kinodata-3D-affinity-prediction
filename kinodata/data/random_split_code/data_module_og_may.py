from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.lightning_datamodule import LightningDataset
#from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.transforms import Compose

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.grouped_split import KinodataKFoldSplit
from sklearn.preprocessing import StandardScaler
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
from torch.utils.data import DataLoader


Kwargs = Dict[str, Any]



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
    dataset_cls: type[InMemoryDataset],
    train_kwargs: Kwargs,
    val_kwargs: Optional[Kwargs] = None,
    test_kwargs: Optional[Kwargs] = None,
    one_time_transform: Optional[Callable[[InMemoryDataset], InMemoryDataset]] = None,
    normalization: bool = False,
    **kwargs,
    ) -> LightningDataset:


    assert_unique_value("pre_transform", train_kwargs, val_kwargs, test_kwargs)

    if split.val_split is not None and val_kwargs is None:
        val_kwargs = deepcopy(train_kwargs)

    if split.test_split is not None and test_kwargs is None:
        test_kwargs = deepcopy(val_kwargs)

    def create_dataset(cls, kwargs, split, ott) -> Optional[InMemoryDataset]:
        if split is None:
            return None
        dataset = cls(**kwargs)
        if ott is not None:
            dataset = ott(dataset)
        return dataset[split]
    
  

    train_dataset = create_dataset(
        dataset_cls, train_kwargs, split.train_split, one_time_transform
    )
    assert train_dataset is not None

    val_dataset = create_dataset(
        dataset_cls, val_kwargs, split.val_split, one_time_transform
    )

    test_dataset = create_dataset(
        dataset_cls, test_kwargs, split.test_split, one_time_transform
    )


    # Normalize function
    def normalize_activity(dataset, scaler):
        if dataset is None:
            return None
        new_data_list = []
        for data in dataset:
            new_data = data.clone()  # Clone to avoid modifying original dataset in-place
            new_data.y = torch.tensor(scaler.transform(data.y.reshape(-1,1)), dtype=torch.float32)[0]
            new_data_list.append(new_data)

        return new_data_list  # Return updated dataset

        
        
    if normalization:

        #normalization
        activity_values = torch.tensor([data.y for data in train_dataset], dtype=torch.float32).reshape(-1,1)

        print("verify normalization of activities")
        print("mean "+str(activity_values.mean().item())+" std "+str(activity_values.std().item()))
        

        scaler = StandardScaler()
        scaler.fit_transform(activity_values.numpy())

        train_dataset = normalize_activity(train_dataset, scaler)
        val_dataset = normalize_activity(val_dataset, scaler)
        test_dataset = normalize_activity(test_dataset, scaler)

        
        normalized_values = torch.tensor([data.y for data in train_dataset], dtype=torch.float32).reshape(-1,1)

        print("**After Normalization**")
        print(f"  Mean: {normalized_values.mean().item():.4f}, Std: {normalized_values.std().item():.4f}")
        print(f"  Sample values: {normalized_values[:5].squeeze().tolist()}")  # Print first few values
        
    


    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
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


def fix_split_for_batch_norm(split: Split, batch_size: int) -> Split:
    def fix(a):
        if len(a) % batch_size == 1:
            a = a[:-1]
        return a

    if split.train_size > 0:
        split.train_split = fix(split.train_split)
    if split.val_size > 0:
        split.val_split = fix(split.val_split)
    if split.test_size > 0:
        split.test_split = fix(split.test_split)

    return split



#trying to treat everything as a large dataset

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl






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
    dataset_cls_2= partial(DavidsdataDocked, remove_hydrogen=config.remove_hydrogen)
    
    dataset_cls_1 = partial(KinodataDocked, remove_hydrogen=config.remove_hydrogen)
    
    

    #dataset_cls ################


    #check this note, not sure if this makes sense anymore?

    if config.filter_rmsd_max_value is not None: #not sure if we want this anymore, since now we may not want to filter by RMSD? only appliying it to the kinodata dataset 
        dataset_cls_2 = Filtered_david(
            dataset_cls_2(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )
        dataset_cls_1 = Filtered(
            dataset_cls_1(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )
  
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


    
    dataset_1 = dataset_cls_1()

    dataset_2 = dataset_cls_2()


    if config.data_split is not None: 
        split = load_precomputed_split(config)
        print("Remapping idents to dataset_1 index..")
        index_mapping_1 = dataset_1.ident_index_map()
        split_1 = split.remap_index(index_mapping_1)
        print(split_1)
        print(f"Split kinodata: Train size {split_1.train_size}, Val size {split_1.val_size}, Test size {split_1.test_size}")
    else:
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        splits_1 = splitter.split(dataset_1)
        split_1 = splits_1[config.split_index]
        #split_1 = splits_1[2]
        print(f"Split kinodata: Train size {split_1.train_size}, Val size {split_1.val_size}, Test size {split_1.test_size}")


    del dataset_1




    if config.data_split is not None:
        split = load_precomputed_split(config)
        print("Remapping idents to dataset_2 index..")
        index_mapping_2 = dataset_2.ident_index_map()
        split_2 = split.remap_index(index_mapping_2)
        print(f"Split rmsd_data: Train size {split_2.train_size}, Val size {split_2.val_size}, Test size {split_2.test_size}")
    else:
       
        #splitter = KinodataKFoldSplit(config.split_type, config.k_fold, max_samples_per_scaffold=800) #should I Put this as a config input? putting here 800 hunderd to diversify validation set when reduced dataset is used
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        print("the type of split is "+str(config.split_type))
        print("the k-fold is "+str(config.k_fold))
        splits_2 = splitter.split(dataset_2)
        split_2 = splits_2[config.split_index]
        print(f"Split rmsd_data: Train size {split_2.train_size}, Val size {split_2.val_size}, Test size {split_2.test_size}")

        ####
        
        #output_dir = Path("../scaffold_splits_check")
            #assert not (unique_train_scaffolds & unique_test_scaffolds), "Overlap between train and test!"
            #assert not (unique_val_scaffolds & unique_test_scaffolds), "Overlap between validation and test!" # test and val are the same, change later!
           
            #total_unique_scaffolds = unique_train_scaffolds | unique_val_scaffolds | unique_test_scaffolds
            #print(f"The total number of unique scaffolds is: {len(total_unique_scaffolds)}")

        #save_scaffolds_to_csv(dataset_2, split_2, output_dir)
    del dataset_2

    
    # dirty batchnorm fix ---DO I NEED THIS?
    split_1 = fix_split_for_batch_norm(split_1, config.batch_size) 
    split_2 = fix_split_for_batch_norm(split_2, config.batch_size) 

    print("Creating data module for kinodataset:")
    print(f"    split:{split_1}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")


    data_module_1 = make_data_module(
        split_1,
        config.batch_size,
        config.num_workers,
        #num_workers = 1,
        dataset_cls=dataset_cls_1,  # type: ignore
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform}, #is this okay?
        one_time_transform=one_time_transform,
        normalization=True
    )

    print("Creating data module for davidsdockeddataset:")
    print(f"    split:{split_2}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    print('the length of the data set is')
    print(dataset_cls_2)

    data_module_2 = make_data_module(
        split_2,
        config.batch_size, 
        config.num_workers,
        #num_workers = 1,
        dataset_cls=dataset_cls_2,  # type: ignore
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform}, #is this okay?
        one_time_transform=one_time_transform,
    )


    print(f"Train size for dataset 1: {split_1.train_size}")
    print(f"Validation size for dataset 1: {split_1.val_size}")
    print(f"Test size for dataset 1: {split_1.test_size}")



    print(f"Train size for dataset 2: {split_2.train_size}")
    print(f"Validation size for dataset 2: {split_2.val_size}")
    print(f"Test size for dataset 2: {split_2.test_size}")


    # Combine both data modules
    #combined_data_module = CombinedDataModule(data_module_1, data_module_2, batch_size=config.batch_size, split_index=0)
    combined_data_module = CombinedDataModule(data_module_1, data_module_2)

    return combined_data_module
