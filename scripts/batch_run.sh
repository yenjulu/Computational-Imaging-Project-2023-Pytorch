#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=20:00:00
#SBATCH --output=run_fastmri_varnet.txt

# python run_fastmri_modl.py
python run_fastmri_varnet.py


#bash scripts/train.sh
#bash scripts/test.sh