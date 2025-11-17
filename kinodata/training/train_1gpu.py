#%%
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from kinodata.data.data_module import make_kinodata_module

import random
import os
import numpy as np


import json
from pathlib import Path
from typing import Any


from functools import partial
import sys

# dirty
sys.path.append(".")
sys.path.append("..")


import torch

import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
#from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph

import kinodata.configuration as configuration
#from kinodata.training import train
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.to_complex_graph import TransformToComplexGraph
import wandb 
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="Fold index for cross-validation")
parser.add_argument("--csv_folder_name", type=str, required=True, help="name of csv folder for output")
parser.add_argument("--checkpoint_name", type=str, required=True, help="name of checkpoint folder")
args = parser.parse_args()

project_name=f"{args.csv_folder_name}_fold_{args.fold}"
print(f"the project name is {project_name}")
#wandb.init(entity="nextaids", project="kinodata-3d_rmsd10", name=project_name, group="iris_kfold_normal_wa_scale_0.2", mode="online", id="cplvjvq2", resume="must", settings=wandb.Settings(silent="false"))
wandb.init(entity="nextaids", project="kinodata-3d_rmsd10", name=project_name, group="pose_weight_homoscedastic_random_wl10_wh7", mode="online", settings=wandb.Settings(silent="false"))



def set_seed(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility
    
    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)
    
    # Environment variable for dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_safe_num_workers():
    try:
        return max(8, len(os.sched_getaffinity(0))) 
    except AttributeError:
        return min(1, os.cpu_count())


####config

configuration.register(
        "sparse_transformer",
        max_num_neighbors=16,
        hidden_channels=256,
        num_attention_blocks=3,
        num_heads=8,
        act="silu",
        edge_attr_size=4,
        ln1=True,
        ln2=True,
        ln3=True,
        graph_norm=False,
        interaction_modes=["covalent", "structural"],
    )


config = configuration.get("data", "training", "sparse_transformer")
config["node_types"] = ["complex"]
config["batch_size"] = 32
config["accumulate_grad_batches"] = 4
config["perturb_complex_positions"] = 0
config["perturb_ligand_positions"] = 0.0
config["need_distances"] = False
config["perturb_pocket_positions"] = 0.0
config["early_stopping_patience"] = 24
config["seed"] = 42 
config["batch_size"] = 32
config["split_type"] = "random-k-fold"
config["filter_rmsd_max_value"] = 10
config["k_fold"] = 5
config["split_index"]=args.fold
config["max_epochs"] = 500
config["accelerator"] = "gpu"
config["csv_save_dir"]=f"/data1/choderaj/lopezrr/kinodata-3D-affinity-prediction/kinodata/training/data_runs_test/{project_name}"
set_seed(config["seed"])
n_w=get_safe_num_workers()
print(f"num workers from train script {n_w}")
config["num_workers"]=n_w

print(f"the configuration is {config}")

#if config["split_type"] == "scaffold-k-fold":
#    from kinodata.data.data_module_scaffold.data_module import make_kinodata_module

#if config["split_type"] == "random-k-fold":
#    from kinodata.data.data_module_random.data_module import make_kinodata_module

#if config["split_type"] == "pocket-k-fold":
#    from kinodata.data.data_module_pocket.data_module import make_kinodata_module


torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
checkpoint_dir = f"{args.checkpoint_name}_fold_{args.fold}"

#to save the model while running
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best_model",
    save_top_k=1,
    monitor="val/combined_mae",  # or "val_activity_loss", etc
    mode="min",
    save_last=True
)

def train(config, fn_data, fn_model=None):
    
    logger = WandbLogger(
        log_model="all",
        )
    
    model = fn_model(config)# Instantiate the model

   
   
    data_module = fn_data
    logger.watch(model, log="all", log_freq=10, log_graph=True)
    print(data_module)

    # Setup the data module to initialize datasets
    data_module.setup(stage='fit')
    
   
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/combined_mae", 
        patience=config.early_stopping_patience, 
        mode="min",
    )
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        max_epochs=config.max_epochs,  
        accelerator=config.accelerator,
        devices=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.clip_grad_value,
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()


    print(f"Max epochs: {trainer.max_epochs}")


    trainer.fit(model, datamodule=data_module)
    #trainer.fit(model, datamodule=data_module, ckpt_path=f"{checkpoint_dir}/last.ckpt")
    trainer.test(ckpt_path="best", datamodule=data_module)


data_module = make_kinodata_module(
    config,
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


train(
        config=config,
        fn_model=make_model,
        fn_data=data_module
    )



model = make_model(config)

wandb.finish()
