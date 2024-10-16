#%%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from kinodata.data.data_module import make_kinodata_module


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
wandb.init(entity="nextaids", project="kinodata_extended", name="cross_entropy_torch_pose_complex_activity", mode="online", settings=wandb.Settings(silent="false"))


#%%
torch.cuda.is_available()

#%%

data_module = make_kinodata_module(
    cfg.get("data", "training").update(
        dict(

            batch_size=32,
            split_type="scaffold-k-fold",
            filter_rmsd_max_value=4.0,
            split_index=0,
        )
    ),
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)


#%%

def train(config, fn_data, fn_model=None):
    logger = WandbLogger(project="kinodata_extended", log_model="all")
    model = fn_model(config)
    data_module = fn_data

    print(data_module)

    # Setup the data module to initialize datasets
    data_module.setup(stage='fit')

        # Print number of batches
    print(f"Number of batches in the current epoch: {len(data_module.train_dataloader())}")

    # Print combined dataset sizes after setup
    # Inside the train function
    print("Training dataset size:", len(data_module.train_dataset))
    print(f"Number of batches per epoch: {len(data_module.train_dataloader())}")

 ###
    print(f"Dataset 1 training (Activity) size: {len(data_module.train_dataset.dataset1)}")
    print(f"Dataset 2 training (Pose) size: {len(data_module.train_dataset.dataset2)}")
    print(f"Dataset 1 val (Activity) size: {len(data_module.val_dataset.dataset1)}")
    print(f"Dataset 2 val (Pose) size: {len(data_module.val_dataset.dataset2)}")
    print(f"Dataset 1 test (Activity) size: {len(data_module.test_dataset.dataset1)}")
    print(f"Dataset 2 test (Pose) size: {len(data_module.test_dataset.dataset2)}")
    


   
    validation_checkpoint = ModelCheckpoint(
        monitor="val/mae",
        mode="min",
    )
    #print(data_module)
    lr_monitor = LearningRateMonitor("epoch")
    early_stopping = EarlyStopping(
        monitor="val/mae", 
    
        patience=config.early_stopping_patience, mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=config.epochs,
        accelerator=config.accelerator,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[validation_checkpoint, lr_monitor, early_stopping],
        gradient_clip_val=config.clip_grad_value,
    )
    if config.dry_run:
        print("Exiting: config.dry_run is set.")
        exit()


    

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
config["need_distances"] = False
config["batch_size"] = 32
config["accumulate_grad_batches"] = 4
config["perturb_complex_positions"] = 0.1

#delete the following two lines for propoer config during training
config["perturb_ligand_positions"] = 0.0
config["perturb_pocket_positions"] = 0.0


print(config)




import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())



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




