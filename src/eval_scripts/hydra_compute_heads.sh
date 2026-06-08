#!/bin/bash
#SBATCH --job-name=fv_heads_MODEL_NICK_DATASET
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclude=head034,head022
#SBATCH --mem=128G
#SBATCH --output=logs/slurm-%j_heads_MODEL_NICK_DATASET.out

source ~/.bashrc
conda activate fv
[[ -f ~/.hf_token ]] && export HF_TOKEN=$(cat ~/.hf_token)
cd REPO_SRC_DIR

python compute_universal_heads.py \
    --model_name MODEL_NAME \
    --save_path_root ../results/MODEL_NICK \
    --datasets DATASET \
    --n_top_heads 100 \
    --n_shots 10 \
    --n_mean_trials 100 \
    --n_ie_trials 25 \
    --seed 42
