#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=32GB           
#PBS -l walltime=00:60:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/EGO-Moment-CLE-ViT/.venv/bin/activate
python3 train.py --config configs/ufg_base.yaml >> out_train_a100.txt

