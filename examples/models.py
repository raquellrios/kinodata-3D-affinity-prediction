#!/usr/bin/env python
# coding: utf-8

# ## Validating pre-trained models
# This code demonstrates how to load our trained models and one way of doing using them for inference.
# It requires that you downloaded and extracted the [pretrained models and the corresponding preprocessed version of kinodata-3D](https://zenodo.org/records/10410594)
# in the root directory of this repository.

# In[1]:


import json
from pathlib import Path
from typing import Any

import torch

import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, DTIModel, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.data.data_module import make_kinodata_module
from kinodata.transform import TransformToComplexGraph


# In[2]:


get_ipython().system('wandb disabled')


# Demo boilerplate code for loading model checkpoints, reuses parts of our training/evaluation code.

# In[3]:


model_dir = Path("..") / "models"
assert model_dir.exists()

def path_to_model(rmsd_threshold: int, split_type: str, split_fold: int, model_type: str) -> Path:
    p = model_dir / f"rmsd_cutoff_{rmsd_threshold}" / split_type / str(split_fold) / model_type
    if not p.exists():
        p.mkdir(parents=True)
    return p
model_cls = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer
}

def load_wandb_config(
    config_file: Path
) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    return config

def load_from_checkpoint(
    rmsd_threshold: int,
    split_type: str,
    fold: int,
    model_type: str
) -> RegressionModel:
    cls = model_cls[model_type]
    p = path_to_model(rmsd_threshold, split_type, fold, model_type)
    model_ckpt = list(p.glob("**/*.ckpt"))[0]
    model_config = p / "config.json"
    ckp = torch.load(model_ckpt, map_location="cpu")
    config = cfg.Config(load_wandb_config(model_config))
    model = cls(config)
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model


# Load model checkpoints for *scaffold-split* data subject to predicted RMSD $\leq 4\text{Å}$, where the $0$-th fold is used as test set.

# In[4]:


cgnn = load_from_checkpoint(4, "scaffold-k-fold", 0, "CGNN")
cgnn_3d = load_from_checkpoint(4, "scaffold-k-fold", 0, "CGNN-3D") 
dti = load_from_checkpoint(4, "scaffold-k-fold", 0, "DTI")


# Create the matching data module

# In[5]:


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


# Fast way of demonstrating inference on just one test batch:

# In[ ]:


demo_test_batch = next(iter(data_module.test_dataloader()))
with torch.no_grad():
    mae_sample = {
        "DTI sample test MAE": dti.test_step(demo_test_batch)["test/mae"],
        "CGNN sample test MAE": cgnn.test_step(demo_test_batch)["test/mae"],
        "CGNN-3D sample test MAE": cgnn_3d.test_step(demo_test_batch)["test/mae"],
    }
mae_sample


# Test all three models using all test data in the current data module

# In[ ]:


from pytorch_lightning import Trainer


# In[ ]:


trainer = Trainer(logger=False)
dti_metrics = trainer.test(model=dti, datamodule=data_module, ckpt_path=None)
cgnn_metrics = trainer.test(model=cgnn, datamodule=data_module, ckpt_path=None)
cgnn_3d_metrics = trainer.test(model=cgnn_3d, datamodule=data_module, ckpt_path=None)


# Display results

# In[ ]:


{
    "DTI": dti_metrics, 
    "CGNN": cgnn_metrics, 
    "CGNN-3D": cgnn_3d_metrics
}

