"""
Generate and submit SLURM jobs for the function-vector pipeline.

Phases
------
compute_heads   Run compute_universal_heads.py for each new model (one job per model).
                After jobs finish, copy the printed top_heads lists into
                extract_utils.py before running the next phase.

layer_sweep     Run evaluate_function_vector.py with --edit_layer -1 for every
                model × dataset combination (one job per pair).
                Inspect the resulting zs_results_layer_sweep.json files to pick
                the best edit_layer for each model, then run fixed_eval.

fixed_eval      Run evaluate_function_vector.py with a specific --edit_layer.
                Requires --edit_layers flag (see usage below).

numheads_sweep  Run test_numheads.py to find the optimal n_top_heads for each
                model × dataset. Requires --edit_layers flag.

Usage
-----
# Phase 1 — compute universal heads for new models
python submit_hydra_jobs.py --phase compute_heads

# Phase 2 — layer sweep for all models × datasets
python submit_hydra_jobs.py --phase layer_sweep

# Phase 3 — fixed-layer eval (after inspecting sweep results)
python submit_hydra_jobs.py --phase fixed_eval \\
    --edit_layers gptj=9,llama32_3b=8,gemma3_4b=12,qwen3_8b=16

# n_heads sweep (find optimal number of heads)
python submit_hydra_jobs.py --phase numheads_sweep \\
    --edit_layers gemma3_4b=11 --models gemma3_4b

# Dry run (print commands without submitting)
python submit_hydra_jobs.py --phase layer_sweep --dry_run
"""
import os
import argparse
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Absolute path to the function_vectors/src directory on the cluster.
# Edit this to match your home directory on the target cluster.
REPO_SRC_DIR = os.path.join(Path(__file__).resolve().parents[2], 'src')

CLUSTERS = {
    'cluster_h': {
        'partition':  'gpu-2d',
        'gpu_flag':   '--gpus-per-node=1',
        'extra':      ['--exclude=head034,head022'],
        'conda_init': 'source ~/.bashrc',
        'conda_env':  'fv',
    },
    'cluster_c': {
        'partition':  'gpu',
        'gpu_flag':   '--gres=gpu:1',
        'extra':      [],
        'conda_init': 'source /etc/profile.d/conda.sh',
        'conda_env':  'function_vectors',
    },
}

PHASE_META = {
    'compute_heads':  {'tpl': 'tpl_compute_heads.sh',  'mem': '128G', 'tag': 'heads'},
    'layer_sweep':    {'tpl': 'tpl_layer_sweep.sh',    'mem': '128G', 'tag': 'sweep'},
    'fixed_eval':     {'tpl': 'tpl_fixed_eval.sh',     'mem': '64G',  'tag': 'eval'},
    'numheads_sweep': {'tpl': 'tpl_test_numheads.sh',  'mem': '64G',  'tag': 'numheads'},
    'filter_check':   {'tpl': 'tpl_filter_check.sh',   'mem': '32G',  'tag': 'filter'},
}

DATASETS = [
    'antonym', 'capitalize', 'country-capital',
    'english-french', 'present-past', 'singular-plural',
]

# Full abstractive task set used for AIE aggregation (compute_heads phase).
# Matches the scope of dataset_files/abstractive/ per the paper's Appendix E approach.
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

MODELS = {
    'EleutherAI/gpt-j-6b': {
        'nick':        'gptj',
        'n_top_heads': 10,
        'new_model':   False,   # universal heads already hardcoded
        'edit_layer':  9,       # 28 layers, L/3
    },
    'meta-llama/Llama-3.2-3B-Instruct': {
        'nick':        'llama32_3b',
        'n_top_heads': 15,
        'new_model':   True,
        'edit_layer':  9,       # 28 layers, L/3
    },
    'google/gemma-3-4b-it': {
        'nick':        'gemma3_4b',
        'n_top_heads': 15,
        'new_model':   True,
        'edit_layer':  11,      # 34 layers, L/3
    },
    'google/gemma-3-4b-pt': {
        'nick':        'gemma3_4b_pt',
        'n_top_heads': 15,
        'new_model':   True,
        'edit_layer':  11,      # 34 layers, L/3
    },
    'Qwen/Qwen3-8B': {
        'nick':        'qwen3_8b',
        'n_top_heads': 25,
        'new_model':   True,
        'edit_layer':  12,      # 36 layers, L/3
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_template(name: str) -> str:
    template_path = Path(__file__).parent / name
    with open(template_path) as f:
        return f.read()


def _fill(template: str, replacements: dict) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace(key, str(value))
    return result


def _make_script(phase: str, cluster: str, replacements: dict) -> str:
    meta = PHASE_META[phase]
    cfg = CLUSTERS[cluster]
    tag = meta['tag']
    sbatch_lines = [
        '#!/bin/bash',
        f'#SBATCH --job-name=fv_{tag}_MODEL_NICK_DATASET',
        f'#SBATCH --partition={cfg["partition"]}',
        f'#SBATCH {cfg["gpu_flag"]}',
        '#SBATCH --ntasks-per-node=4',
        f'#SBATCH --mem={meta["mem"]}',
        f'#SBATCH --output=logs/slurm-%j_{tag}_MODEL_NICK_DATASET.out',
    ] + [f'#SBATCH {e}' for e in cfg['extra']]
    conda = f'{cfg["conda_init"]}\nconda activate {cfg["conda_env"]}'
    body = _read_template(meta['tpl'])
    script = '\n'.join(sbatch_lines) + f'\n\n{conda}\n{body}'
    return _fill(script, replacements)


def _submit(script_content: str, script_path: str, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_content)
    cmd = f"sbatch {script_path}"
    if dry_run:
        print(f"[DRY RUN] {cmd}")
    else:
        print(f"Submitting: {cmd}")
        os.system(cmd)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_compute_heads(job_dir: str, dry_run: bool, cluster: str, model_filter: set = None, dataset_filter: set = None) -> None:
    for model_name, cfg in MODELS.items():
        if not cfg['new_model']:
            continue
        nick = cfg['nick']
        if model_filter and nick not in model_filter:
            continue
        for dataset in ABSTRACTIVE_DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            script = _make_script('compute_heads', cluster, {
                'MODEL_NICK':   nick,
                'MODEL_NAME':   model_name,
                'DATASET':      dataset,
                'REPO_SRC_DIR': REPO_SRC_DIR,
            })
            path = os.path.join(job_dir, f'compute_heads_{nick}_{dataset}.sh')
            _submit(script, path, dry_run)
            print(f"  -> {path}")


def phase_layer_sweep(job_dir: str, dry_run: bool, cluster: str, model_filter: set = None, dataset_filter: set = None) -> None:
    for model_name, cfg in MODELS.items():
        nick = cfg['nick']
        if model_filter and nick not in model_filter:
            continue
        for dataset in DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            script = _make_script('layer_sweep', cluster, {
                'MODEL_NICK':   nick,
                'MODEL_NAME':   model_name,
                'DATASET':      dataset,
                'N_TOP_HEADS':  cfg['n_top_heads'],
                'REPO_SRC_DIR': REPO_SRC_DIR,
            })
            path = os.path.join(job_dir, f'sweep_{nick}_{dataset}.sh')
            _submit(script, path, dry_run)
            print(f"  -> {path}")


def phase_fixed_eval(job_dir: str, edit_layers: dict, dry_run: bool, cluster: str, model_filter: set = None, dataset_filter: set = None) -> None:
    for model_name, cfg in MODELS.items():
        nick = cfg['nick']
        if model_filter and nick not in model_filter:
            continue
        layer = edit_layers.get(nick, cfg['edit_layer'])
        for dataset in DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            script = _make_script('fixed_eval', cluster, {
                'MODEL_NICK':   nick,
                'MODEL_NAME':   model_name,
                'DATASET':      dataset,
                'N_TOP_HEADS':  cfg['n_top_heads'],
                'EDIT_LAYER':   layer,
                'REPO_SRC_DIR': REPO_SRC_DIR,
            })
            path = os.path.join(job_dir, f'eval_{nick}_{dataset}_layer{layer}.sh')
            _submit(script, path, dry_run)
            print(f"  -> {path}")


def phase_filter_check(job_dir: str, dry_run: bool, cluster: str, model_filter: set = None, dataset_filter: set = None) -> None:
    """Submit one lightweight job per (model, dataset) to fill in missing filter JSONs."""
    for model_name, cfg in MODELS.items():
        if not cfg['new_model']:
            continue
        nick = cfg['nick']
        if model_filter and nick not in model_filter:
            continue
        for dataset in ABSTRACTIVE_DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            script = _make_script('filter_check', cluster, {
                'MODEL_NICK':   nick,
                'MODEL_NAME':   model_name,
                'DATASET':      dataset,
                'REPO_SRC_DIR': REPO_SRC_DIR,
            })
            path = os.path.join(job_dir, f'filter_{nick}_{dataset}.sh')
            _submit(script, path, dry_run)
            print(f"  -> {path}")


def phase_numheads_sweep(job_dir: str, edit_layers: dict, dry_run: bool, cluster: str, model_filter: set = None, dataset_filter: set = None) -> None:
    for model_name, cfg in MODELS.items():
        nick = cfg['nick']
        if model_filter and nick not in model_filter:
            continue
        layer = edit_layers.get(nick, cfg['edit_layer'])
        for dataset in DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            script = _make_script('numheads_sweep', cluster, {
                'MODEL_NICK':   nick,
                'MODEL_NAME':   model_name,
                'DATASET':      dataset,
                'N_TOP_HEADS':  cfg['n_top_heads'],
                'EDIT_LAYER':   layer,
                'REPO_SRC_DIR': REPO_SRC_DIR,
            })
            path = os.path.join(job_dir, f'numheads_{nick}_{dataset}.sh')
            _submit(script, path, dry_run)
            print(f"  -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_edit_layers(s: str) -> dict:
    """Parse 'gptj=9,llama32_3b=8' into {'gptj': 9, 'llama32_3b': 8}."""
    result = {}
    for pair in s.split(','):
        nick, layer = pair.strip().split('=')
        result[nick.strip()] = int(layer.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True,
                        choices=['compute_heads', 'layer_sweep', 'fixed_eval', 'numheads_sweep', 'filter_check'])
    parser.add_argument('--edit_layers', type=str, default='',
                        help='Comma-separated nick=layer pairs, e.g. gptj=9,llama32_3b=8')
    parser.add_argument('--models', type=str, default='',
                        help='Comma-separated model nicks to run (default: all). e.g. gptj,llama32_3b')
    parser.add_argument('--datasets', type=str, default='',
                        help='Comma-separated dataset names to run (default: all). e.g. antonym,english-french')
    parser.add_argument('--cluster', choices=list(CLUSTERS), default='cluster_h',
                        help='Target cluster (default: cluster_h)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print sbatch commands without running them')
    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    job_dir = os.path.join(Path(__file__).parent, 'cluster_jobs', timestamp)
    os.makedirs(job_dir, exist_ok=True)

    # Ensure logs directory exists relative to REPO_SRC_DIR
    logs_dir = os.path.join(REPO_SRC_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Phase: {args.phase}")
    print(f"Cluster: {args.cluster}")
    print(f"Job scripts: {job_dir}")
    print(f"Dry run: {args.dry_run}\n")

    model_filter  = set(args.models.split(','))  if args.models  else set()
    dataset_filter = set(args.datasets.split(',')) if args.datasets else set()

    if args.phase == 'compute_heads':
        phase_compute_heads(job_dir, args.dry_run, args.cluster, model_filter, dataset_filter)

    elif args.phase == 'layer_sweep':
        phase_layer_sweep(job_dir, args.dry_run, args.cluster, model_filter, dataset_filter)

    elif args.phase == 'fixed_eval':
        edit_layers = parse_edit_layers(args.edit_layers) if args.edit_layers else {}
        phase_fixed_eval(job_dir, edit_layers, args.dry_run, args.cluster, model_filter, dataset_filter)

    elif args.phase == 'filter_check':
        phase_filter_check(job_dir, args.dry_run, args.cluster, model_filter, dataset_filter)

    elif args.phase == 'numheads_sweep':
        edit_layers = parse_edit_layers(args.edit_layers) if args.edit_layers else {}
        phase_numheads_sweep(job_dir, edit_layers, args.dry_run, args.cluster, model_filter, dataset_filter)

    total = len(list(Path(job_dir).glob('*.sh')))
    print(f"\nSubmitted {total} job(s).")


if __name__ == '__main__':
    main()
