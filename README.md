# Multi-objective Kinodata3D

Code and data-processing utilities for **No Pose Left Behind: Integrating Activity and Structural Data with Uncertainty-Aware Multiobjective Learning for Kinase Inhibitor Prediction**.

This repository contains the implementation of `mOKDDD`, a multi-objective E(3)-invariant graph neural network for kinase–ligand binding affinity prediction. The model is designed to learn from experimentally measured activity data paired with *in silico*-generated kinase–ligand complex structures of varying pose quality.

Unlike structure-based models that rely only on poses below a fixed RMSD cutoff, `mOKDDD` explicitly models structural reliability. For each kinase–ligand complex, the model jointly predicts:

- binding affinity, expressed as pIC50;
- activity uncertainty;
- pose quality, expressed as a continuous structural reliability score.

The predicted pose quality is used to modulate the contribution of each structure to the activity loss, allowing the model to learn from heterogeneous structural data while giving greater importance to reliable complexes.

## Overview
![Schematic overview of the multi-objective model workflow](methods_detailed_fig.png)

Structure-based machine learning for kinase inhibitor prediction is limited by the scarcity of experimentally resolved protein–ligand complexes. Computationally generated structures, such as docked poses, can reduce this limitation, but their usefulness depends strongly on pose quality.

`mOKDDD` addresses this by combining two training objectives:

1. **Activity prediction objective**  
   Learns binding affinity and activity uncertainty from kinase–ligand complexes with experimental activity labels.

2. **Pose-quality objective**  
   Learns to estimate the structural reliability of generated ligand poses using RMSD-derived pose-quality labels.

Both objectives are optimized jointly using a shared E(3)-invariant message-passing GNN and a multi-output readout.

## Model outputs

For each kinase–ligand complex `x`, the model predicts:

```text
mu(x)       predicted activity / binding affinity
sigma(x)    predicted activity uncertainty
q_pose(x)   predicted pose quality / structural reliability

## Installation

We currently support installation from source.

### 1. Clone this repository

```
git clone https://github.com/raquellrios/multi-objective-kinodata-3D.git
cd multi-objective-kinodata-3D
git checkout paper_release
```

### 2. Set up Python environment
You can use mamba or conda to set up the environment
```
mamba env create -f kinodata_env.yml
mamba activate kinodata_env
```

Then, install the package with

```
pip install -e .
```