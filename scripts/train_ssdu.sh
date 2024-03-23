#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:50:00

GPU_NUM=0

TRAIN_CONFIG_YAML="configs/fastmri_ssdu.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train_modl_ssdu.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=3 \

# CUDA_VISIBLE_DEVICES=$GPU_NUM python train_torch.py \
#     --config=$TRAIN_CONFIG_YAML \
#     --write_image=3 \