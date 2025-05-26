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


#%%
import wandb 


# Initialize wandb with settings to ensure logging
wandb.finish()

project_name="x2_pose_lilac_logits_normalised_act_soft_kfold1_clean_reg_datamod_test"
#project_name= "random_split_kfold1_soft_learnable_rmsd_shift_clamp_5_set_act_normalised"
#project_name="model_wact0_wpose1_pose_scaffold_all_data_soft_rmsd_split1:5_scaffold_tests_perturb_positions_0.2"
#project_name="x2_pose_data_iris_logits_normalised_act_soft_kfold1"
wandb.init(entity="nextaids", project="kinodata-3d_rmsd10", name=project_name, mode="online", settings=wandb.Settings(silent="false"))


#%%
print(torch.cuda.is_available())
print(torch.version.cuda)  # PyTorch CUDA version
print(torch.backends.cudnn.version())  # cuDNN version
print(torch.cuda.get_device_name(0))  # Should match your GPU

#%%

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

# Call this function at the start of your script
set_seed(41)

##

data_module = make_kinodata_module(
    cfg.get("data", "training").update(
        dict(

            batch_size=32,
            split_type="random-k-fold",
            filter_rmsd_max_value=10,
            split_index=0,
        )
    ),
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


#%%



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
    
   
    validation_checkpoint = ModelCheckpoint(
        monitor="val/mae_activity",
        mode="min",
    )
    #print(data_module)
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/mae_activity", 
        patience=config.early_stopping_patience, 
        mode="min",
        
        #enabled=False #only changing this to see the model overfitting, change back for normal training!
    )
    
    #USE_ONE_FORWARD = False
    #model.use_one_forward = USE_ONE_FORWARD

    #if USE_ONE_FORWARD:
    #   print("Using one-forward method.")
    #else:
    #   print("Using two-forward method.")

    trainer = pl.Trainer(
        logger=logger,
        #auto_select_gpus=True,
        max_epochs=500,   #config.epochs,  
        accelerator=config.accelerator,
        devices=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[validation_checkpoint, lr_monitor, early_stopping],
        #callbacks=[validation_checkpoint, lr_monitor], #only for overfit
        gradient_clip_val=config.clip_grad_value,
        #overfit_batches=2


    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()


    print(f"Max epochs: {trainer.max_epochs}")

    print(f"Config value for perturb_ligand_positions inside the train function: {config['perturb_ligand_positions']}")


    trainer.fit(model, datamodule=data_module)
    #trainer.test(ckpt_path="best", datamodule=data_module)

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
####
config["perturb_complex_positions"] = 0
config["perturb_ligand_positions"] = 0.0
config["need_distances"] = False
config["perturb_pocket_positions"] = 0.0
####
config["early_stopping_patience"] = 100
config["seed"] = 16
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
config["split_index"]=0

#get the right workers
if os.cpu_count != 1:
    n_w=os.cpu_count() -2
else:
    n_w=1
print(f"num workers from train script {n_w}")
config["num_workers"]=n_w


config["csv_save_dir"]="/data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/data_runs/x2_pose_logits_normalised_act_soft_kfold1_cleanscript_reg_datamod_datasets_test"
print(config)




import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())



print(f"Config value for perturb_ligand_positions before calling make_kinodata_module: {config['perturb_ligand_positions']}")


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
