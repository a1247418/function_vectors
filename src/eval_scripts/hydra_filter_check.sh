#!/bin/bash
#SBATCH --job-name=fv_filter_MODEL_NICK_DATASET
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclude=head034,head022
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j_filter_MODEL_NICK_DATASET.out

source ~/.bashrc
conda activate fv
[[ -f ~/.hf_token ]] && export HF_TOKEN=$(cat ~/.hf_token)
cd REPO_SRC_DIR

python compute_universal_heads.py \
    --model_name MODEL_NAME \
    --datasets DATASET \
    --save_path_root ../results/MODEL_NICK \
    --root_data_dir ../dataset_files \
    --filter_only \
    --n_shots 10 \
    --seed 42
