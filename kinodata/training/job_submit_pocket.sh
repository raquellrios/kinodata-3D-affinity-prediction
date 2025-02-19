#!/bin/bash
#BSUB -P "create-data_KINODATA"
#BSUB -J "one_forward"
#BSUB -R span[hosts=1]
#BSUB -oo /data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training/pocket_kfold2.out
#BSUB -cwd /data/chodera/lopezrir/kinodata-3D-affinity-prediction/kinodata/training 
#BSUB -n 1
#BSUB -R rusage[mem=256]
#BSUB -q gpuqueue
#BSUB -gpu "num=1:gmem=42G"
#BSUB -W 30:40
# Load your bash profile to ensure conda is initialized
source ~/.bashrc

conda activate kinodata_backup 

python3 train_pocket.py
