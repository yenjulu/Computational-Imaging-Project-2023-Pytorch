#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=10:00:00
#SBATCH --output=log.txt

python fastMRI_to_modl_dataset.py