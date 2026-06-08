"""
Print a task-performance table from fixed-layer eval results.

Usage (from src/):
    python eval_scripts/summarize_results.py --results_root ../results

Looks for files matching:
    <results_root>/<model_nick>/<dataset>/zs_results_editlayer_*.json
    <results_root>/<model_nick>/<dataset>/fs_shuffled_results_editlayer_*.json
    <results_root>/<model_nick>/<dataset>/model_baseline.json
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


def summarize(results_root):
    model_nicks = sorted(
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    )

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
