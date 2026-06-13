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
