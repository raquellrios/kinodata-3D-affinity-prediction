#!/bin/bash

#SBATCH --partition gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:07:00
#SBATCH --job-name=norm_act_with_logits
#SBATCH --output=%j_%x_%N.out
#SBATCH --error=%j_%x_%N.err
#SBATCH --mem-per-cpu=150G
#SBATCH --gpus-per-task=1

# Source .bashrc -- has to be done manually so far
source ${HOME}/.bashrc


conda activate kinodata_backup



python3 train.py
