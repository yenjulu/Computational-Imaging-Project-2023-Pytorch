#!/bin/bash
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=23:00:00

GPU_NUM=0
# TEST_CONFIG_YAML="configs/base_modl.yaml"
# TEST_CONFIG_YAML="configs/fastmri_modl.yaml"

# TEST_CONFIG_YAML="configs/base_varnet.yaml"
# TEST_CONFIG_YAML="configs/fastmri_varnet.yaml"
TEST_CONFIG_YAML="configs/fastmri_ssdu_test.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test_modl_ssdu.py \
    --config=$TEST_CONFIG_YAML \
    --write_image=10 \
    # --batch_size=32