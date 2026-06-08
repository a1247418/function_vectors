"""
Print per-layer zero-shot accuracy from layer sweep results (evaluate_function_vector.py --edit_layer -1).

Usage (from src/):
    python eval_scripts/summarize_layer_sweep.py --results_root ../results
    python eval_scripts/summarize_layer_sweep.py --results_root ../results --model gptj
    python eval_scripts/summarize_layer_sweep.py --results_root ../results --model gemma3_4b --dataset antonym
"""
import os, json, glob, argparse
import numpy as np

DATASETS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']


def top1(rank_list):
    return round(100 * sum(r == 0 for r in rank_list) / len(rank_list), 1) if rank_list else float('nan')


def load_sweep(path):
    with open(path) as f:
        data = json.load(f)
    return {int(k): top1(v.get('intervention_rank_list', [])) for k, v in data.items()}


def summarize_model(model_dir, model_nick, datasets, out_lines):
    # Collect per-dataset per-layer accuracy
    all_layers = set()
    dataset_accs = {}

    for dataset in datasets:
        pattern = os.path.join(model_dir, dataset, 'zs_results_layer_sweep.json')
        matches = glob.glob(pattern)
        if not matches:
            continue
        accs = load_sweep(matches[0])
        dataset_accs[dataset] = accs
        all_layers.update(accs.keys())

    if not dataset_accs:
        return

    layers = sorted(all_layers)
    col_w = 16
    present_datasets = list(dataset_accs.keys())

    header = f"{'Layer':<8}" + "".join(f"{d:>{col_w}}" for d in present_datasets) + f"{'AVERAGE':>{col_w}}"
    separator = '-' * len(header)

    out_lines.append(f"\n{'='*len(header)}")
    out_lines.append(f"Model: {model_nick}  —  Zero-Shot + FV accuracy % per layer")
    out_lines.append(header)
    out_lines.append(separator)

    best_avg = -1
    best_layer = None

    for layer in layers:
        row = f"{layer:<8}"
        vals = []
        for dataset in present_datasets:
            acc = dataset_accs[dataset].get(layer, float('nan'))
            vals.append(acc)
            row += f"{acc:>{col_w}}"
        valid = [v for v in vals if not np.isnan(v)]
        avg = round(np.mean(valid), 1) if valid else float('nan')
        row += f"{avg:>{col_w}}"
        out_lines.append(row)
        if not np.isnan(avg) and avg > best_avg:
            best_avg = avg
            best_layer = layer

    out_lines.append(separator)
    out_lines.append(f"Best layer: {best_layer}  (avg {best_avg}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='../results')
    parser.add_argument('--model', default=None, help='Restrict to one model nick')
    parser.add_argument('--dataset', default=None, help='Restrict to one dataset')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS

    if args.model:
        model_nicks = [args.model]
    else:
        model_nicks = sorted(
            d for d in os.listdir(args.results_root)
            if os.path.isdir(os.path.join(args.results_root, d))
        )

    out_lines = []
    for nick in model_nicks:
        model_dir = os.path.join(args.results_root, nick)
        summarize_model(model_dir, nick, datasets, out_lines)

    output = "\n".join(out_lines)
    print(output)

    out_path = os.path.join(args.results_root, 'layer_sweep_summary.txt')
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
