#!/bin/bash
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=23:50:00

GPU_NUM=0
# TRAIN_CONFIG_YAML="configs/base_modl.yaml"
# TRAIN_CONFIG_YAML="configs/base_varnet.yaml"

TRAIN_CONFIG_YAML="configs/fastmri_modl.yaml"
# TRAIN_CONFIG_YAML="configs/fastmri_varnet.yaml"

# TRAIN_CONFIG_YAML="configs/fastmri_modl_trans.yaml"
# TRAIN_CONFIG_YAML="configs/fastmri_varnet_trans.yaml"

# TRAIN_CONFIG_YAML="configs/fastmri_varnet_U_T.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=1 \

# TRAIN_CONFIG_YAML="configs/fastmri_ssdu.yaml"

# CUDA_VISIBLE_DEVICES=$GPU_NUM python train_torch.py \
#     --config=$TRAIN_CONFIG_YAML \
#     --write_image=10 \