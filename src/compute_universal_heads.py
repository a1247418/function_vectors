"""
Compute a model's universal head ranking by aggregating indirect effects across datasets.

For each dataset this script loads or computes:
  1. mean head activations  (Phase 1 of the FV pipeline)
  2. indirect effect scores (Phase 2 of the FV pipeline)

It then averages the per-dataset indirect effect tensors across all datasets and
prints the ranked (layer, head, score) list ready to copy-paste into the
`compute_universal_function_vector` function in utils/extract_utils.py.

Following the paper (Todd et al. 2024), the AIE is aggregated over all abstractive
tasks for which 10-shot ICL top-1 accuracy exceeds the majority-label baseline.
Pass --no_filter to skip this criterion and include all datasets.

Usage:
    python compute_universal_heads.py \
        --model_name Qwen/Qwen3-8B \
        --save_path_root ../results/qwen3_8b \
        --n_top_heads 100

Cached .pt files are reused if present, so individual dataset jobs can be
pre-run with compute_average_activations.py / compute_indirect_effect.py first.
"""
import os, json, argparse
from collections import Counter
import torch
import numpy as np

from utils.prompt_utils import load_dataset
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.extract_utils import get_mean_head_activations
from utils.eval_utils import n_shot_eval_no_intervention
from compute_indirect_effect import compute_indirect_effect

# All abstractive datasets available under dataset_files/abstractive/.
# These are the tasks used for AIE aggregation, matching the paper's Appendix E scope.
ABSTRACTIVE_DATASETS = [
    'ag_news', 'antonym', 'capitalize', 'capitalize_first_letter',
    'capitalize_last_letter', 'capitalize_second_letter', 'commonsense_qa',
    'country-capital', 'country-currency', 'english-french', 'english-german',
    'english-spanish', 'landmark-country', 'lowercase_first_letter',
    'lowercase_last_letter', 'national_parks', 'next_capital_letter',
    'next_item', 'park-country', 'person-instrument', 'person-occupation',
    'person-sport', 'present-past', 'prev_item', 'product-company',
    'sentiment', 'singular-plural', 'synonym', 'word_length',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name')
    parser.add_argument('--datasets', nargs='+', default=ABSTRACTIVE_DATASETS,
                        help='Datasets to aggregate over (default: all abstractive datasets)')
    parser.add_argument('--no_filter', action='store_true',
                        help='Skip the ICL > majority-label filter; include all datasets')
    parser.add_argument('--filter_only', action='store_true',
                        help='Only run the ICL filter check and cache results; skip mean activations, IE, and aggregation')
    parser.add_argument('--root_data_dir', type=str, default='../dataset_files')
    parser.add_argument('--save_path_root', type=str, default='../results',
                        help='Root directory; per-dataset subdirs are created automatically')
    parser.add_argument('--n_top_heads', type=int, default=100,
                        help='Number of top heads to include in the printed list')
    parser.add_argument('--n_shots', type=int, default=10)
    parser.add_argument('--n_mean_trials', type=int, default=100,
                        help='Trials for mean activation estimation')
    parser.add_argument('--n_ie_trials', type=int, default=25,
                        help='Trials for indirect effect estimation')
    parser.add_argument('--test_split', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prefixes', type=json.loads,
                        default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument('--separators', type=json.loads,
                        default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument('--revision', type=str, default=None)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    print(f"Loading model: {args.model_name}")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    set_seed(args.seed)

    all_ie = []

    for dataset_name in args.datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        save_dir = os.path.join(args.save_path_root, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        dataset = load_dataset(dataset_name, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)

        # Phase 1 — mean head activations (skipped in filter_only mode)
        ma_path = os.path.join(save_dir, f'{dataset_name}_mean_head_activations.pt')
        if not args.filter_only:
            if os.path.exists(ma_path):
                print(f"  Loading cached mean activations from {ma_path}")
                mean_activations = torch.load(ma_path)
            else:
                print(f"  Computing mean activations ({args.n_mean_trials} trials)...")
                mean_activations = get_mean_head_activations(
                    dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                    n_icl_examples=args.n_shots, N_TRIALS=args.n_mean_trials,
                    prefixes=args.prefixes, separators=args.separators,
                )
                torch.save(mean_activations, ma_path)
                print(f"  Saved to {ma_path}")

        # Phase 2 — indirect effects
        ie_path = os.path.join(save_dir, f'{dataset_name}_indirect_effect.pt')

        # Filter: 10-shot ICL top-1 must exceed majority-label baseline (paper criterion).
        # Result is cached in a JSON file so re-runs skip the expensive ICL evaluation.
        # If an IE file already exists it was computed after the filter passed — also skip.
        # Mean activations are always saved (needed for task-specific FV injection regardless).
        filter_path = os.path.join(save_dir, f'{dataset_name}_filter.json')
        if not args.no_filter and not os.path.exists(ie_path):
            if os.path.exists(filter_path):
                with open(filter_path) as f:
                    fdata = json.load(f)
                passes = fdata['passes']
                print(f"  Filter (cached): 10-shot={fdata['top1_acc']:.3f}  majority={fdata['majority_frac']:.3f}  → {'INCLUDE' if passes else 'SKIP'}")
            else:
                outputs = dataset['valid']['output']
                outputs_flat = [o[0] if isinstance(o, list) else str(o) for o in outputs]
                majority_frac = Counter(outputs_flat).most_common(1)[0][1] / len(outputs_flat)
                icl_results = n_shot_eval_no_intervention(
                    dataset, n_shots=args.n_shots, model=model, model_config=model_config,
                    tokenizer=tokenizer, prefixes=args.prefixes, separators=args.separators,
                    compute_ppl=False, test_split='valid',
                )
                top1_acc = dict(icl_results['clean_topk'])[1]
                passes = top1_acc > majority_frac
                with open(filter_path, 'w') as f:
                    json.dump({'passes': bool(passes), 'top1_acc': float(top1_acc), 'majority_frac': float(majority_frac)}, f)
                print(f"  Filter: 10-shot={top1_acc:.3f}  majority={majority_frac:.3f}  → {'INCLUDE' if passes else 'SKIP'}")
            if not passes:
                continue

        if args.filter_only:
            continue  # JSON saved above; skip IE and aggregation for this dataset

        if os.path.exists(ie_path):
            print(f"  Loading cached indirect effects from {ie_path}")
            indirect_effect = torch.load(ie_path)
        else:
            print(f"  Computing indirect effects ({args.n_ie_trials} trials)...")
            indirect_effect = compute_indirect_effect(
                dataset, mean_activations, model=model, model_config=model_config,
                tokenizer=tokenizer, n_shots=args.n_shots, n_trials=args.n_ie_trials,
                last_token_only=True, prefixes=args.prefixes, separators=args.separators,
            )
            torch.save(indirect_effect, ie_path)
            print(f"  Saved to {ie_path}")

        # indirect_effect shape: (n_trials, n_layers, n_heads) -> mean over trials
        all_ie.append(indirect_effect.mean(dim=0))

    if args.filter_only:
        print("\nFilter-only run complete.")
        return

    if not all_ie:
        print("\nNo datasets passed the ICL filter. Run with --no_filter to debug.")
        return

    # Aggregate across datasets
    avg_ie = torch.stack(all_ie).mean(dim=0)  # (n_layers, n_heads)
    n_layers, n_heads = avg_ie.shape

    ranked = sorted(
        [(l, h, round(avg_ie[l, h].item(), 4))
         for l in range(n_layers) for h in range(n_heads)],
        key=lambda x: x[2],
        reverse=True,
    )

    top = ranked[:args.n_top_heads]
    model_nick = args.model_name.split('/')[-1]
    n_included = len(all_ie)

    print("\n" + "=" * 70)
    print(f"# Top {args.n_top_heads} heads for {model_nick}")
    print(f"# Aggregated over: {n_included}/{len(args.datasets)} datasets (passed ICL filter)")
    print(f"# Paste this block into compute_universal_function_vector()")
    print(f"# in src/utils/extract_utils.py, replacing the empty top_heads = []")
    print("=" * 70)

    # Format as compact rows of 10 tuples per line for readability
    row_size = 10
    rows = [top[i:i + row_size] for i in range(0, len(top), row_size)]
    indent = " " * 16
    lines = [", ".join(str(t) for t in r) for r in rows]
    head_str = (",\n" + indent).join(lines)

    print(f"\nelif '{model_nick}' in model_config['name_or_path']:")
    print(f"    top_heads = [{head_str}]")
    print()


if __name__ == '__main__':
    main()
