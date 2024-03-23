#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=04:00:00
#SBATCH --output=top_run.txt

python run_modl.py