#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --output=top_run_varnet.txt

python run_varnet.py