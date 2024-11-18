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
#wandb.init(entity="nextaids", project="kinodata_extended", name="distribution_plot", mode="online", settings=wandb.Settings(silent="false"))


#%%
torch.cuda.is_available()

#%%

data_module = make_kinodata_module(
    cfg.get("data", "training").update(
        dict(

            batch_size=32,
            split_type="scaffold-k-fold",
            filter_rmsd_max_value=10,
            split_index=0,
        )
    ),
    transforms=[TransformToComplexGraph(remove_heterogeneous_representation=False)],
)

