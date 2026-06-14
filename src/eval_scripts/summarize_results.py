"""
Print a task-performance table from fixed-layer eval results.

Usage (from src/):
    python eval_scripts/summarize_results.py --results_root ../results

Looks for files matching:
    <results_root>/<model_nick>/<dataset>/zs_results_editlayer_*.json
    <results_root>/<model_nick>/<dataset>/fs_shuffled_results_editlayer_*.json
    <results_root>/<model_nick>/<dataset>/model_baseline.json

Also prints BL-FV sections (best-layer baseline) from layer sweep files:
    <results_root>/<model_nick>/<dataset>/zs_results_layer_sweep.json
    <results_root>/<model_nick>/<dataset>/fs_shuffled_results_layer_sweep.json
"""
import os, json, glob, argparse
import numpy as np

DATASETS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']


def top1(rank_list):
    return round(100 * sum(r == 0 for r in rank_list) / len(rank_list), 1) if rank_list else float('nan')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_fixed_layer_file(directory, prefix):
    pattern = os.path.join(directory, f'{prefix}_editlayer_*.json')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None  # take the most recent if multiple


def find_best_layer(model_dir, datasets):
    """Return the layer with highest mean ZS accuracy across datasets (from layer sweep)."""
    layer_avgs = {}
    for dataset in datasets:
        path = os.path.join(model_dir, dataset, 'zs_results_layer_sweep.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            acc = top1(v.get('intervention_rank_list', []))
            if not np.isnan(acc):
                layer_avgs.setdefault(int(k), []).append(acc)
    if not layer_avgs:
        return None
    return max(layer_avgs, key=lambda l: np.mean(layer_avgs[l]))


def acc_at_layer(model_dir, dataset, sweep_prefix, layer, key='intervention_rank_list'):
    """Read accuracy for a specific layer from a layer-sweep JSON."""
    if layer is None:
        return float('nan')
    path = os.path.join(model_dir, dataset, f'{sweep_prefix}.json')
    if not os.path.exists(path):
        return float('nan')
    with open(path) as f:
        data = json.load(f)
    entry = data.get(str(layer)) or data.get(layer)
    return top1(entry.get(key, [])) if entry else float('nan')


def summarize(results_root):
    model_nicks = sorted(
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    )

    # Pre-compute best injection layer per model from layer sweep results
    best_layers = {
        nick: find_best_layer(os.path.join(results_root, nick), DATASETS)
        for nick in model_nicks
    }

    col_w = 16
    header = f"{'Dataset':<20}" + "".join(f"{n:>{col_w}}" for n in model_nicks)

    lines = []

    for label, prefix, key in [
        ("Zero-Shot + FV (accuracy %)",     "zs_results",          "intervention_rank_list"),
        ("Shuffled-Label + FV (accuracy %)", "fs_shuffled_results", "intervention_rank_list"),
        ("ICL Baseline (accuracy %)",        "model_baseline",      "clean_rank_list"),
    ]:
        lines.append(f"\n{'='*len(header)}")
        lines.append(label)
        lines.append(header)
        lines.append('-' * len(header))

        accs = {nick: [] for nick in model_nicks}

        for dataset in DATASETS:
            row = f"{dataset:<20}"
            for nick in model_nicks:
                d = os.path.join(results_root, nick, dataset)
                if prefix == "model_baseline":
                    fpath = os.path.join(d, 'model_baseline.json')
                else:
                    fpath = find_fixed_layer_file(d, prefix)

                if fpath and os.path.exists(fpath):
                    data = load_json(fpath)
                    if prefix == "model_baseline":
                        # keyed by shot count; use the highest available
                        shot_key = str(max(int(k) for k in data.keys()))
                        ranks = data[shot_key].get(key, [])
                    else:
                        ranks = data.get(key, [])
                    acc = top1(ranks)
                    accs[nick].append(acc)
                    row += f"{acc:>{col_w}}"
                else:
                    accs[nick].append(float('nan'))
                    row += f"{'—':>{col_w}}"
            lines.append(row)

        lines.append('-' * len(header))
        avg_row = f"{'AVERAGE':<20}"
        for nick in model_nicks:
            vals = [v for v in accs[nick] if not np.isnan(v)]
            avg_row += f"{round(np.mean(vals), 1) if vals else float('nan'):>{col_w}}"
        lines.append(avg_row)

    # BL-FV sections: inject at the best layer found by the layer sweep
    for label, sweep_prefix, key in [
        ("BL-FV Zero-Shot (accuracy %)",      "zs_results_layer_sweep",         "intervention_rank_list"),
        ("BL-FV Shuffled-Label (accuracy %)", "fs_shuffled_results_layer_sweep", "intervention_rank_list"),
    ]:
        lines.append(f"\n{'='*len(header)}")
        lines.append(label)
        lines.append(header)
        lines.append('-' * len(header))

        accs = {nick: [] for nick in model_nicks}

        for dataset in DATASETS:
            row = f"{dataset:<20}"
            for nick in model_nicks:
                model_dir = os.path.join(results_root, nick)
                acc = acc_at_layer(model_dir, dataset, sweep_prefix, best_layers[nick], key)
                accs[nick].append(acc)
                row += f"{acc:>{col_w}}" if not np.isnan(acc) else f"{'—':>{col_w}}"
            lines.append(row)

        lines.append('-' * len(header))
        avg_row = f"{'AVERAGE':<20}"
        for nick in model_nicks:
            vals = [v for v in accs[nick] if not np.isnan(v)]
            bl = best_layers[nick]
            label_suffix = f"L={bl}" if bl is not None else "no sweep"
            avg_val = round(np.mean(vals), 1) if vals else float('nan')
            cell = f"{avg_val} ({label_suffix})"
            avg_row += f"{cell:>{col_w}}"
        lines.append(avg_row)

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(results_root, 'summary.txt')
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='../results',
                        help='Root results directory containing per-model subdirs')
    args = parser.parse_args()
    summarize(args.results_root)


if __name__ == '__main__':
    main()
