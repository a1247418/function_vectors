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
