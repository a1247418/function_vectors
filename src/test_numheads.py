import argparse
import json
import os
import numpy as np
import torch

from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset
from utils.extract_utils import compute_universal_function_vector

# Evaluates how performance changes as the number of heads used to create a Function Vector increases
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help="dataset to be evaluated", type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--model_nickname', type=str, required=False, default='gptj')
    parser.add_argument('--n_heads', type=int, help="upper bound of the number of heads to create the FV", required=True, default=40)
    parser.add_argument('--edit_layer', type=int, help="layer at which to add the function vector", required=True, default=9)
    parser.add_argument('--seed', required=False, type=int, default=42)
    parser.add_argument('--save_path_root', required=True, type=str, default='../results',
                        help='Root results dir; mean activations expected at save_path_root/<dataset_name>/<dataset_name>_mean_head_activations.pt')
    parser.add_argument('--root_data_dir', type=str, required=False, default='../dataset_files')
    parser.add_argument('--prefixes', type=json.loads, required=False,
                        default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument('--separators', type=json.loads, required=False,
                        default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument('--device', type=str, required=False,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--revision', type=str, required=False, default=None)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    dataset = load_dataset(args.dataset_name, root_data_dir=args.root_data_dir, seed=args.seed)

    ma_path = os.path.join(args.save_path_root, args.dataset_name,
                           f'{args.dataset_name}_mean_head_activations.pt')
    mean_activations = torch.load(ma_path)
    print(f"Loaded mean activations from {ma_path}")

    set_seed(args.seed)
    fs_results = n_shot_eval_no_intervention(
        dataset, n_shots=10, model=model, model_config=model_config, tokenizer=tokenizer,
        prefixes=args.prefixes, separators=args.separators,
    )
    filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
    print(f"Sanity check — 10-shot topk: {fs_results['clean_topk']}, filter_set size: {len(filter_set)}")

    zs_results = {}
    for i in range(1, args.n_heads + 1):
        fv, _ = compute_universal_function_vector(mean_activations, model, model_config, i)
        result = n_shot_eval(
            dataset, fv, args.edit_layer, 0, model, model_config, tokenizer,
            filter_set=filter_set, prefixes=args.prefixes, separators=args.separators,
        )
        zs_results[i] = result
        topk = result.get('intervention_topk', result.get('topk', '?'))
        print(f"  n_heads={i:3d}  intervention_topk={topk}")

    out_dir = os.path.join(args.save_path_root, f'{args.model_nickname}_test_numheads')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset_name}_perf_v_heads.json')
    json.dump(zs_results, open(out_path, 'w'))
    print(f"\nSaved to {out_path}")
