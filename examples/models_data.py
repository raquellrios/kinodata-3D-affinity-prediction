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


#get_ipython().system('wandb disabled')


# Demo boilerplate code for loading model checkpoints, reuses parts of our training/evaluation code.

# In[3]:


# Load model checkpoints for *scaffold-split* data subject to predicted RMSD $\leq 4\text{Å}$, where the $0$-th fold is used as test set.

# In[4]:


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

