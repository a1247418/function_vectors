#!/bin/bash
#SBATCH --job-name=fv_numheads_MODEL_NICK_DATASET
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclude=head034,head022
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j_numheads_MODEL_NICK_DATASET.out

source ~/.bashrc
conda activate fv
[[ -f ~/.hf_token ]] && export HF_TOKEN=$(cat ~/.hf_token)
cd REPO_SRC_DIR

python test_numheads.py \
    --model_name MODEL_NAME \
    --model_nickname MODEL_NICK \
    --dataset_name DATASET \
    --save_path_root ../results/MODEL_NICK \
    --edit_layer EDIT_LAYER \
    --n_heads N_TOP_HEADS \
    --seed 42
