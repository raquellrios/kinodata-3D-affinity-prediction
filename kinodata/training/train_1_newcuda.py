#%%
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
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph

import kinodata.configuration as configuration
#from kinodata.training import train
from kinodata.model.complex_transformer import ComplexTransformer, make_model
from kinodata.types import NodeType
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform.to_complex_graph import TransformToComplexGraph
import wandb 
from pytorch_lightning.callbacks import ModelCheckpoint

# wandb initialization
wandb.finish()

project_name="iris_32cpus_normal_arithmetic_learnweights_const_0.5_fixed"
#wandb.init(entity="nextaids", project="kinodata-3d_rmsd10", name=project_name, mode="online", id="d2eyqe1y", resume="must", settings=wandb.Settings(silent="false"))
wandb.init(entity="nextaids", project="kinodata-3d_rmsd10", name=project_name, mode="online", settings=wandb.Settings(silent="false"))
#checking cuda stuff
print(torch.cuda.is_available())
print(torch.version.cuda)  # PyTorch CUDA version
print(torch.backends.cudnn.version())  # cuDNN version
print(torch.cuda.get_device_name(0))  # Should match your GPU


torch.set_float32_matmul_precision('high')   # or 'medium' for a bit more precision


def set_seed(seed=41):
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

# setting seed
set_seed(41)


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
config["seed"] = 16 #probably need ot change this so it mathces the seed 41
config["batch_size"] = 32
config["split_type"] = "random-k-fold"
config["filter_rmsd_max_value"] = 10
config["split_index"]=0
config["max_epochs"] = 500
config["accelerator"] = "gpu"

def get_safe_num_workers():
    try:
        return max(8, len(os.sched_getaffinity(0))) 
    except AttributeError:
        return min(1, os.cpu_count())
n_w=get_safe_num_workers()

print(f"num workers from train script {n_w}")
config["num_workers"]=n_w
config["csv_save_dir"]="/data1/choderaj/lopezrr/kinodata-3D-affinity-prediction/kinodata/training/data_runs/iris_32cpus_normal_arithmetic_learnweights_const_05_fixed"


print(f"the configuration is {config}")
####
#parameters for pose
#config["weight_decay"] = 0.005n
#config["dropout"] = 0.2
#for overfitting delete afterwards
#parameters below for overfit
#config["lr"] = 1e-3
#config["min_lr"] = 1e-4
#config["max_epochs"] = 600
#smaller model
#config["hidden_channels"] = 64
#config["num_attention_blocks"] = 2
#config["num_heads"]=2

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

#to save the model while running
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints_normal_arithmetic_learnweights_const_05_fixed/",
    filename="best_model",
    save_top_k=1,
    monitor="val/combined_mae",  # or "val_activity_loss", etc
    mode="min",
    save_last=True
)

def train(config, fn_data, fn_model=None):
    
    logger = WandbLogger(
        project="kinodata-3d_rmsd10",
        log_model="all",
        )
    
    model = fn_model(config)# Instantiate the model

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params}")    
   
    data_module = fn_data
    logger.watch(model, log="all", log_freq=10, log_graph=True)
    print(data_module)

    # Setup the data module to initialize datasets
    data_module.setup(stage='fit')
    
   
    #validation_checkpoint = ModelCheckpoint(
    #    monitor="val/mae_activity",
    #    mode="min",
    #)
    #print(data_module)
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/combined_mae", 
        patience=config.early_stopping_patience, 
        mode="min",
    )
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        #auto_select_gpus=True,
        max_epochs=config.max_epochs,  
        accelerator=config.accelerator,
        #strategy = "ddp",
        devices=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        #callbacks=[validation_checkpoint, lr_monitor, early_stopping],
        gradient_clip_val=config.clip_grad_value,
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()


    print(f"Max epochs: {trainer.max_epochs}")

    trainer.fit(model, datamodule=data_module)
    #trainer.fit(model, datamodule=data_module, ckpt_path="checkpoints_normal_wa_scale/last.ckpt")
    #trainer.test(ckpt_path="best", datamodule=data_module)


data_module = make_kinodata_module(
    config,
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


train(
        config=config,
        fn_model=make_model,
        fn_data=data_module
        #partial(
            #make_kinodata_module,
            #data_module,
            #one_time_transform=partial(
            #    apply_transform_instance_permament,
            #    transform=TransformToComplexGraph(
            #        remove_heterogeneous_representation=True
            #    ),
            #),
        #),
    )


#%%
