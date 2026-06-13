[[ -f ~/.hf_token ]] && export HF_TOKEN=$(cat ~/.hf_token)
cd REPO_SRC_DIR

python evaluate_function_vector.py \
    --dataset_name DATASET \
    --model_name MODEL_NAME \
    --save_path_root ../results/MODEL_NICK \
    --n_top_heads N_TOP_HEADS \
    --edit_layer EDIT_LAYER \
    --universal_set \
    --seed 42
