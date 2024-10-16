#!/bin/bash
#BSUB -P "create-data_KINODATA"
#BSUB -J "only_data_creation"
#BSUB -R span[hosts=1]
#BSUB -oo /data/chodera/lopezrir/kinodata-3D-affinity-prediction/examples/combined_output.out
#BSUB -cwd /data/chodera/lopezrir/kinodata-3D-affinity-prediction/examples 
#BSUB -n 1
#BSUB -R rusage[mem=32]
#BSUB -q gpuqueue
#BSUB -gpu "num=1:gmem=11G"
#BSUB -W 24:00
# Load your bash profile to ensure conda is initialized
source ~/.bashrc

conda activate kinodata_gpu_lilac

python3 models_data.py
