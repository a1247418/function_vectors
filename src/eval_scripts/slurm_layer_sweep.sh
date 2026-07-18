#!/bin/bash
#SBATCH --job-name=fv_sweep_MODEL_NICK_DATASET
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --output=logs/slurm-%j_sweep_MODEL_NICK_DATASET.out

source ~/.bashrc
conda activate fv
[[ -f ~/.hf_token ]] && export HF_TOKEN=$(cat ~/.hf_token)
cd REPO_SRC_DIR

python evaluate_function_vector.py \
    --dataset_name DATASET \
    --model_name MODEL_NAME \
    --save_path_root ../results/MODEL_NICK \
    --n_top_heads N_TOP_HEADS \
    --edit_layer -1 \
    --universal_set \
    --seed 42
