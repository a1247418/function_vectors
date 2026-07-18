"""
Summarize test_numheads.py sweep results (n_heads vs ZS+FV accuracy).

Reads every <dataset>_perf_v_heads.json in the given results directory and prints:
  - per dataset: top-1 accuracy at each swept head count, with the best marked
  - aggregate: mean top-1 across datasets per head count, with the best n_heads

Usage (from src/):
    python summarize_numheads.py --results_dir ../results/qwen3_8b_test_numheads
    python summarize_numheads.py --results_dir ../results/qwen3_8b_test_numheads --step 1
"""
import argparse
import glob
import json
import os

import numpy as np


def top1_curve(path):
    """Return {n_heads: top1} for one sweep JSON."""
    with open(path) as f:
        data = json.load(f)
    curve = {}
    for n_str, result in data.items():
        topk = result.get('intervention_topk') or result.get('topk')
        curve[int(n_str)] = dict((int(k), v) for k, v in topk)[1]
    return dict(sorted(curve.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='e.g. ../results/qwen3_8b_test_numheads')
    parser.add_argument('--step', type=int, default=5,
                        help='Print every STEP-th head count in the per-dataset tables (default 5)')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, '*_perf_v_heads.json')))
    if not files:
        raise SystemExit(f"No *_perf_v_heads.json files in {args.results_dir}")

    curves = {}
    for f in files:
        ds = os.path.basename(f).replace('_perf_v_heads.json', '')
        curves[ds] = top1_curve(f)

    all_n = sorted(set().union(*[set(c) for c in curves.values()]))
    common_n = sorted(set.intersection(*[set(c) for c in curves.values()]))
    name_w = max(len(d) for d in curves)

    # --- per-dataset tables ---
    shown_n = [n for n in all_n if n % args.step == 0 or n == 1 or n == all_n[-1]]
    print(f"{len(curves)} datasets, n_heads swept over [{all_n[0]}..{all_n[-1]}]\n")
    print("Top-1 accuracy vs n_heads (* = best for that dataset):")
    print(f"  {'dataset':<{name_w}}  " + "".join(f"{n:>7}" for n in shown_n) + f"  {'best':>12}")
    for ds, c in sorted(curves.items()):
        best_n = max(c, key=c.get)
        row = f"  {ds:<{name_w}}  "
        for n in shown_n:
            row += f"{c[n]:>7.3f}" if n in c else f"{'-':>7}"
        row += f"  {c[best_n]:>6.3f} @{best_n:<3d}"
        print(row)

    # --- aggregate over datasets ---
    print(f"\nMean top-1 across all {len(curves)} datasets (common n range [{common_n[0]}..{common_n[-1]}]):")
    mean_curve = {n: float(np.mean([curves[d][n] for d in curves])) for n in common_n}
    best_n = max(mean_curve, key=mean_curve.get)
    for n in [n for n in common_n if n % args.step == 0 or n == 1 or n == common_n[-1]]:
        bar = '#' * round(mean_curve[n] * 40)
        mark = '  <-- best' if n == best_n else ''
        print(f"  n={n:3d}  mean_top1={mean_curve[n]:.4f}  {bar}{mark}")

    top5 = sorted(mean_curve.items(), key=lambda kv: -kv[1])[:5]
    print("\nBest 5 head counts by mean top-1: " + ", ".join(f"n={n} ({v:.4f})" for n, v in top5))
