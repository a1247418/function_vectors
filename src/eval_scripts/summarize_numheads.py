"""
Summarize test_numheads.py results: accuracy vs n_heads, averaged across datasets.

Usage (from src/):
    python eval_scripts/summarize_numheads.py --results_root ../results --model_nick gemma3_4b
    python eval_scripts/summarize_numheads.py --results_root ../results --model_nick gptj --dataset antonym
"""
import os, json, glob, argparse
import numpy as np

DATASETS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']


def top1(rank_list):
    return round(100 * sum(r == 0 for r in rank_list) / len(rank_list), 1) if rank_list else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='../results')
    parser.add_argument('--model_nick', required=True, help='Model nickname, e.g. gemma3_4b')
    parser.add_argument('--dataset', default=None, help='Restrict to one dataset')
    args = parser.parse_args()

    sweep_dir = os.path.join(args.results_root, args.model_nick, f'{args.model_nick}_test_numheads')
    if not os.path.isdir(sweep_dir):
        print(f"No numheads results found at {sweep_dir}")
        return

    datasets = [args.dataset] if args.dataset else DATASETS

    # Load per-dataset results; key = n_heads (int), value = top-1 accuracy
    dataset_accs = {}
    for dataset in datasets:
        path = os.path.join(sweep_dir, f'{dataset}_perf_v_heads.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        dataset_accs[dataset] = {
            int(k): top1(v.get('intervention_rank_list', []))
            for k, v in raw.items()
        }

    if not dataset_accs:
        print(f"No JSON files found in {sweep_dir}")
        return

    all_n = sorted({n for accs in dataset_accs.values() for n in accs})
    present = list(dataset_accs.keys())

    col_w = 16
    header = f"{'n_heads':<10}" + "".join(f"{d:>{col_w}}" for d in present) + f"{'AVERAGE':>{col_w}}"
    sep = '-' * len(header)

    print(f"\nModel: {args.model_nick}  —  Zero-Shot + FV accuracy % vs n_heads")
    print(header)
    print(sep)

    best_avg, best_n = -1, None
    rows = []
    for n in all_n:
        vals = [dataset_accs[d].get(n, float('nan')) for d in present]
        valid = [v for v in vals if not np.isnan(v)]
        avg = round(np.mean(valid), 1) if valid else float('nan')
        row = f"{n:<10}" + "".join(f"{v:>{col_w}}" for v in vals) + f"{avg:>{col_w}}"
        rows.append(row)
        if not np.isnan(avg) and avg > best_avg:
            best_avg, best_n = avg, n

    print("\n".join(rows))
    print(sep)
    print(f"Best n_heads: {best_n}  (avg {best_avg}%)")

    out_path = os.path.join(args.results_root, f'{args.model_nick}_numheads_summary.txt')
    with open(out_path, 'w') as f:
        f.write(header + "\n" + sep + "\n")
        f.write("\n".join(rows) + "\n")
        f.write(sep + "\n")
        f.write(f"Best n_heads: {best_n}  (avg {best_avg}%)\n")
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
