from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.loader.dataloader import DataLoader
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


#### analyse scaffolds

def save_scaffolds_to_csv(dataset, splits, output_dir):
    """
    Saves the specific scaffolds for each split into a CSV file.
    """
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    for i, split in enumerate(splits):
        output_file = output_dir / f"scaffold_split_fold_{i+1}.csv"
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Set', 'Scaffold'])
            for idx in split.train_index:
                writer.writerow([idx, 'Train', dataset[idx].scaffold])
            for idx in split.val_index:
                writer.writerow([idx, 'Validation', dataset[idx].scaffold])
            for idx in split.test_index:
                writer.writerow([idx, 'Test', dataset[idx].scaffold])

        print(f"Scaffolds for Fold {i+1} saved to {output_file}")

def print_scaffolds_in_splits(dataset, csv_file):
        """
        Print the scaffolds in each split.
        """

        # Load the CSV
        split_df = pd.read_csv(csv_file)
        
        # Create a mapping of splits to identifiers
        train_idents = split_df[split_df["split"] == "train"]["ident"].tolist()
        val_idents = split_df[split_df["split"] == "val"]["ident"].tolist()
        test_idents = split_df[split_df["split"] == "test"]["ident"].tolist()
        
        # Extract scaffolds for each split
        train_scaffolds = [dataset[idx].scaffold for idx in train_idents]
        val_scaffolds = [dataset[idx].scaffold for idx in val_idents]
        test_scaffolds = [dataset[idx].scaffold for idx in test_idents]
        
        # Print scaffold information
        print("Train Scaffolds:", set(train_scaffolds))
        print("Validation Scaffolds:", set(val_scaffolds))
        print("Test Scaffolds:", set(test_scaffolds))

def save_scaffolds_to_csv(dataset, csv_file, output_dir: Path):
        """
        Save the scaffolds and their split assignments to CSV files.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Load the CSV
        split_df = pd.read_csv(csv_file)
        
        # Create mappings for splits
        for split_name in ["train", "val", "test"]:
            split_idents = split_df[split_df["split"] == split_name]["ident"].tolist()
            split_scaffolds = [dataset[idx].scaffold for idx in split_idents]

            # Save the scaffolds to a CSV file
            output_path = os.path.join(output_dir, f"{split_name}_scaffolds.csv")
            pd.DataFrame({"scaffold": split_scaffolds}).to_csv(output_path, index=False)
            print(f"Saved scaffolds for {split_name} to {output_path}")


def count_scaffold_distribution(dataset, csv_file):
        """
        Count the occurrences of each scaffold in the dataset for each split.
        """
        # Load the CSV
        split_df = pd.read_csv(csv_file)

        # Initialize a dictionary to store scaffold counts
        scaffold_counts = {}

        # Count scaffolds for each split
        for split_name in ["train", "val", "test"]:
            split_idents = split_df[split_df["split"] == split_name]["ident"].tolist()
            split_scaffolds = [dataset[idx].scaffold for idx in split_idents]
            scaffold_counts[split_name] = pd.Series(split_scaffolds).value_counts()

        return scaffold_counts


def visualise_scaffold_overlap_with_bars(dataset, csv_file):
    """
    Visualise scaffold overlap across splits using a bar chart.
    
    Args:
        dataset: The dataset object with `scaffold` attributes.
        csv_file: Path to the CSV file containing the `ident` and `split` columns.
        save_path: Path to save the bar chart figure (optional).
    """
    # Load the CSV
    split_df = pd.read_csv(csv_file)

    # Extract scaffolds for each split
    scaffolds = {}
    for split_name in ["train", "val", "test"]:
        split_idents = split_df[split_df["split"] == split_name]["ident"].tolist()
        scaffolds[split_name] = set([dataset[idx].scaffold for idx in split_idents])

    # Count unique and overlapping scaffolds
    train_only = len(scaffolds["train"] - scaffolds["val"] - scaffolds["test"])
    val_only = len(scaffolds["val"] - scaffolds["train"] - scaffolds["test"])
    test_only = len(scaffolds["test"] - scaffolds["train"] - scaffolds["val"])
    train_val = len(scaffolds["train"] & scaffolds["val"] - scaffolds["test"])
    train_test = len(scaffolds["train"] & scaffolds["test"] - scaffolds["val"])
    val_test = len(scaffolds["val"] & scaffolds["test"] - scaffolds["train"])
    all_three = len(scaffolds["train"] & scaffolds["val"] & scaffolds["test"])

    # Create bar chart
    labels = [
        "Train Only", "Validation Only", "Test Only",
        "Train & Validation", "Train & Test", "Validation & Test", "All Three"
    ]
    counts = [train_only, val_only, test_only, train_val, train_test, val_test, all_three]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color="skyblue")
    plt.title("Scaffold Overlap Across Splits")
    plt.ylabel("Number of Scaffolds")
    plt.xticks(rotation=45, ha="right")
    
    plt.title("Scaffold Overlap Across Splits")
    plt.savefig("../scaffold_splits_check/scaffold_overlab.png", format='png', dpi=300, bbox_inches='tight')


##############


def make_data_module(
    split: Split,
    batch_size: int,
    num_workers: int = 1,
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
from torch.utils.data import Dataset, DataLoader #, CombinedLoader
#from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
#from pytorch_lightning.trainer.supporters import CombinedLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl


#class CombinedDataset(Dataset):
#    def __init__(self, dataset1, dataset2):
#        """
#        dataset1: Dataset A (Activity dataset)
#        dataset2: Dataset B (Pose dataset)
#        """
#        self.dataset1 = dataset1
#        self.dataset2 = dataset2
#        self.len1 = len(dataset1)
#        self.len2 = len(dataset2)
#        # length is the larger dataset
#        self.max_len = max(len(dataset1), len(dataset2))
#        #self.min_len = min(self.len1, self.len2)#
#
#        print("len of kinodata is "+str(self.len1))
#        print("len of davids_data is "+str(self.len2))
#        #print("the len of each dataset must then be "+str(self.min_len))


#    def __len__(self):
#        return self.max_len  # Adjust this if you want equal-size batches or oversampling

#    def __getitem__(self, idx):
#        """
#        Returns a combined batch containing activity and pose data
#        """
#        item1 = self.dataset1[idx]
#        item2 = self.dataset2[idx]#

        #print("the len of kinodata inside the combineddataset is "+str(len(item1)))
        #print("the len of davidsdata inside the combineddataset is "+str(len(item2)))

        #should I change the above so that it is like this?
        # Ensure same index is used for both datasets
        #idx = idx % self.min_len
        #item1 = self.dataset1[idx ]  # Wrap around if lengths differ
        #item2 = self.dataset2[idx ]  # Wrap around if lengths differ

#        return {'activity_batch': item1, 'pose_batch': item2}


#def custom_collate(batch):
#    """
#    Custom collate function to handle activity and pose batches separately.
#    """
#    activity_batches = [item['activity_batch'] for item in batch]
#    pose_batches = [item['pose_batch'] for item in batch]#

#    # Create PyTorch Geometric batches from activity and pose data
#    activity_batch = Batch.from_data_list(activity_batches)
#    pose_batch = Batch.from_data_list(pose_batches)#

#    #print("inside the custom_collate the kinodata batch is "+str(len(activity_batches)))
#    #print("inside the custom_collate the davids batch is "+str(len(pose_batches)))

#    return activity_batch, pose_batch


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, dm_a, dm_b):
        super().__init__()
        self.dm_a = dm_a  # Data module for activity dataset
        self.dm_b = dm_b  # Data module for pose dataset
        

    def setup(self, stage=None):
        self.dm_a.setup(stage=stage)
        self.dm_b.setup(stage=stage)





    print(CombinedLoader.__init__.__doc__)

    # def train_dataloader(self):
    #     return [self.dm_a.train_dataloader(), self.dm_b.train_dataloader()]

    # def val_dataloader(self):
    #     return [self.dm_a.val_dataloader(), self.dm_b.val_dataloader()]
    
    # def test_dataloader(self):
    #     return [self.dm_a.test_dataloader(), self.dm_b.test_dataloader()]



    #def train_dataloader(self):
    #    return {"activity": self.dm_a.train_dataloader(), "pose": self.dm_b.train_dataloader()}

    #def val_dataloader(self):
    #    return {"activity": self.dm_a.val_dataloader(), "pose": self.dm_b.val_dataloader()}

    #def test_dataloader(self):
    #    return {"activity": self.dm_a.test_dataloader(), "pose": self.dm_b.test_dataloader()}

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




#class CombinedDataModule(pl.LightningDataModule):
#    def __init__(self, datamodule1, datamodule2, batch_size, split_index, num_workers=1):
#        """
#        CombinedDataModule for handling two datasets (activity and pose).
#        
#        datamodule1: DataModule for Activity dataset (Dataset A)
#        datamodule2: DataModule for Pose dataset (Dataset B)
#        """
#        super().__init__()
#        self.datamodule1 = datamodule1
#        self.datamodule2 = datamodule2
#        self.batch_size = batch_size
#        self.num_workers = num_workers
#        self.split_index= split_index
##
#
#    def setup(self, stage=None):##
#
#        """
#        Setup function to combine training, validation, and test datasets.
#        """##
#
#        from torch.utils.data import Subset#
#
#        # Combine training datasets
#        self.train_dataset = CombinedDataset(self.datamodule1.train_dataset, self.datamodule2.train_dataset)##
#
#        # Combine validation datasets
#        self.val_dataset = CombinedDataset(self.datamodule1.val_dataset, self.datamodule2.val_dataset)#
#        #self.val_dataset = CombinedDataset(self.datamodule1.train_dataset, self.datamodule2.train_dataset) #for overfit####
#
#        # Combine test datasets
#        self.test_dataset = CombinedDataset(self.datamodule1.test_dataset, self.datamodule2.test_dataset)
#        
#        #davids data after transformation
#        import matplotlib.pyplot as plt
#        import numpy as np 


        #subset_train= self.datamodule2.train_dataset
        #rmsd_davids = torch.tensor([data.predicted_rmsd for data in subset_train])
        #rmsd_filtered_davids = rmsd_davids[rmsd_davids <= 10].cpu().numpy()
        #rmsd_filtered_davids_transform = 1 / (1 + np.exp( 0.7 * (rmsd_filtered_davids - 4.5)))
        #plt.figure(figsize=(8, 6))
        #plt.hist(rmsd_filtered_davids_transform, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
        #plt.title('RMSD Distribution after sigmoid (0 to 1)')
        #plt.xlabel('RMSD prob')
        #plt.ylabel('Frequency')
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.savefig("../davids_rmsd_distrib_prob_transform_after_process_training.png", format='png', dpi=300, bbox_inches='tight')
        #plt.close()


        #subset_val= self.datamodule2.val_dataset
        #rmsd_davids = torch.tensor([data.predicted_rmsd for data in subset_val])
        #rmsd_filtered_davids = rmsd_davids[rmsd_davids <= 10].cpu().numpy()
        #rmsd_filtered_davids_transform = 1 / (1 + np.exp( 0.7 * (rmsd_filtered_davids - 4.5)))
        #plt.figure(figsize=(8, 6))
        #plt.hist(rmsd_filtered_davids_transform, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
        #plt.title('RMSD Distribution after sigmoid (0 to 1)')
        #plt.xlabel('RMSD prob')
        #plt.ylabel('Frequency')
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.savefig("../davids_rmsd_distrib_prob_transform_after_process_validation.png", format='png', dpi=300, bbox_inches='tight')
        #plt.close()




#    def train_dataloader(self):
#        return DataLoader(
#            self.train_dataset, 
#            batch_size=self.batch_size, 
#            num_workers=self.num_workers, 
#            pin_memory=False, 
#            drop_last=False,
#            collate_fn=custom_collate,#
#	        shuffle=True #adding this following joschka's recommendation -- false for overfit
#        )

#    def val_dataloader(self):
#        return DataLoader(
#            self.val_dataset, 
#            batch_size=self.batch_size, 
#            num_workers=self.num_workers, 
#            pin_memory=False, 
#            collate_fn=custom_collate
#        )

#    def test_dataloader(self):
#        return DataLoader(
#            self.test_dataset, 
#            batch_size=self.batch_size, 
#            num_workers=self.num_workers, 
#            pin_memory=False, 
#            collate_fn=custom_collate
#        )

####


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

    ########################################plotting RMSD distributions

    import matplotlib.pyplot as plt

    kinodata3d_activity=torch.tensor([data.y for data in dataset_1])

    range_max=10
    
    bins=50

    plt.figure(figsize=(8, 6))
    plt.hist(kinodata3d_activity, bins=bins, range=(0, 20), color='skyblue', edgecolor='black')
    plt.title('Activity Distribution (0 to 20)')
    plt.xlabel('pIC50')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("../kinodata3d_activity_distrib.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

    rmsd_kinodata3d = torch.tensor([data.predicted_rmsd for data in dataset_1])
    rmsd_davids = torch.tensor([data.predicted_rmsd for data in dataset_2])


    rmsd_filtered_kinodata3d = rmsd_kinodata3d[rmsd_kinodata3d <= range_max].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_filtered_kinodata3d, bins=bins, range=(0, range_max), color='skyblue', edgecolor='black')
    plt.title('RMSD Distribution (0 to 20)')
    plt.xlabel('RMSD')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("../kinodata3d_rmsd_distrib.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

    rmsd_filtered_davids = rmsd_davids[rmsd_davids <= range_max].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_filtered_davids, bins=bins, range=(0, range_max), color='skyblue', edgecolor='black')
    plt.title('RMSD Distribution (0 to 20)')
    plt.xlabel('RMSD')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("../davids_rmsd_distrib.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
   
   #davids data after transformation
    import numpy as np 
    rmsd_filtered_davids = rmsd_davids[rmsd_davids <= range_max].cpu().numpy()
    rmsd_filtered_davids_transform = 1 / (1 + np.exp( 0.6 * (rmsd_filtered_davids - 4)))
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_filtered_davids_transform, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
    plt.title('RMSD Distribution after sigmoid (0 to 1)')
    plt.xlabel('RMSD prob')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("../davids_rmsd_distrib_prob_transform.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

   ###################### scaffold analysis

    #csv_file= "/lila/data/chodera/lopezrir/kinodata-3D-affinity-prediction/data/processed/davidsdocked/filter_predicted_rmsd_le10.00/scaffold-k-fold/1:5.csv"
    #csv_file= "/lila/data/chodera/lopezrir/kinodata-3D-affinity-prediction/data/processed/davidsdocked/filter_predicted_rmsd_le10.00/scaffold-k-fold/5:5.csv"


    #print_scaffolds_in_splits(dataset_2, csv_file)
    #output_dir = Path("../scaffold_splits_check")
    #save_scaffolds_to_csv(dataset_2, csv_file, output_dir)

    #scaffold_counts = count_scaffold_distribution(dataset_2, csv_file)
    #for split_name, counts in scaffold_counts.items():
    #    print(f"{split_name.capitalize()} Scaffold Counts:")
    #    print(counts)

    #visualise_scaffold_overlap_with_bars(dataset_2, csv_file)



   #######

    if config.data_split is not None: #do dataset1 and dataset 2 need to be splitted in the samew way??? ASK
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
        #print("running this with config.data_split is not None!!!!!!!")
        split = load_precomputed_split(config)
        print("Remapping idents to dataset_2 index..")
        index_mapping_2 = dataset_2.ident_index_map()
        split_2 = split.remap_index(index_mapping_2)
        #print(split_2)
        print(f"Split rmsd_data: Train size {split_2.train_size}, Val size {split_2.val_size}, Test size {split_2.test_size}")
    else:
        #print("running this with config.data_split None!!!!!!!")
        #splitter = KinodataKFoldSplit(config.split_type, config.k_fold, max_samples_per_scaffold=800) #should I Put this as a config input? putting here 800 hunderd to diversify validation set when reduced dataset is used
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        print("the type of split is "+str(config.split_type))
        print("the k-fold is "+str(config.k_fold))
        splits_2 = splitter.split(dataset_2)
        split_2 = splits_2[config.split_index]
        #split_2 = splits_2[2]
        print(split_2)
        print(f"Split rmsd_data: Train size {split_2.train_size}, Val size {split_2.val_size}, Test size {split_2.test_size}")

        ####
        output_dir = Path("../scaffold_splits_check")

        for i, split in enumerate(splits_2, start=1):
            print(f"Split {i}:")
            train_scaffolds = [dataset_2[idx].scaffold for idx in split.train_split]
            unique_train_scaffolds = set(train_scaffolds)
            val_scaffolds = [dataset_2[idx].scaffold for idx in split.val_split]
            unique_val_scaffolds = set(val_scaffolds)
            test_scaffolds = [dataset_2[idx].scaffold for idx in split.test_split]
            unique_test_scaffolds = set(test_scaffolds)
            print(f"  Unique Train Scaffolds: {len(set(train_scaffolds))}")
            print(f"  Unique Validation Scaffolds: {len(set(val_scaffolds))}")
            print(f"   Unique Test Scaffolds: {len(set(test_scaffolds))}")

            #assertion rules:
            #assert not (unique_train_scaffolds & unique_val_scaffolds), "Overlap between train and validation!"
            

            counts={i:val_scaffolds.count(i) for i in val_scaffolds}

            print("the count of the scaffolds is")
            print(counts)

            rmsds= []

            for idx in split.val_split:

                if dataset_2[idx].scaffold == "C1CCC2C(C1)CCC2C1CCCC1":

                    rmsds.append((dataset_2[idx].predicted_rmsd).tolist())

            rmsds = [
                    x
                    for xs in rmsds
                    for x in xs
                ]

            #rmsds=torch.stack(rmsds, dim=0).tolist()

            print(rmsds)

            if len(rmsds) > 2:

                
                # Plot histograms for each transformation
                
                rmsd_transformed =  [1 / (1 + np.exp(0.7 * (x - 4.5))) for x in rmsds]
            
                    # Create the histogram
            plt.figure(figsize=(8, 6))
            plt.hist(rmsd_transformed, bins=10, range=(0, 1))
            plt.title(f'Training RMSD Distribution  of C1CCC2C(C1)CCC2C1CCCC1' )
            plt.xlabel('RMSD prob')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
                    # Save the histogram as a file
            filename = f"../rmsd_C1CCC2C(C1)CCC2C1CCCC1.png"
            plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
            plt.close()
      





            #assert not (unique_train_scaffolds & unique_test_scaffolds), "Overlap between train and test!"
            #assert not (unique_val_scaffolds & unique_test_scaffolds), "Overlap between validation and test!" # test and val are the same, change later!

            #if i == 3:
            #    print("for i=3")
            #    print(set(val_scaffolds))

            #if i == 1:
            #    print("for i=1")
            #    print(set(val_scaffolds))
            

            total_unique_scaffolds = unique_train_scaffolds | unique_val_scaffolds | unique_test_scaffolds
            print(f"The total number of unique scaffolds is: {len(total_unique_scaffolds)}")


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
        #config.num_workers,
        num_workers = 1,
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
        #config.num_workers,
        num_workers = 1,
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
