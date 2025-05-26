#!/bin/bash
#BSUB -P "create-data_KINODATA"
#BSUB -J "one_forward"
#BSUB -R span[hosts=1]
#BSUB -oo /data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/normalised_act_bce_logits_shift_learnable_regularization.out
#BSUB -cwd /data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training 
#BSUB -n 1
#BSUB -R rusage[mem=256]
#BSUB -q gpuqueue
#BSUB -gpu "num=1:gmem=42G"
#BSUB -W 48:00
# Load your bash profile to ensure conda is initialized
source ~/.bashrc

conda activate kinodata_backup

python3 train.py
