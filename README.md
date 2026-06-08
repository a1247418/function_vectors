# Function Vectors in Large Language Models (fork)

> **This is a personal fork** of [ericwtodd/function_vectors](https://github.com/ericwtodd/function_vectors), the official implementation of the ICLR 2024 paper [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213). Extended to support additional models (Gemma-3, Qwen3, Llama-3.2) and a SLURM batch workflow.

## Setup

We recommend using conda as a package manager.
The environment used for this project can be found in the `fv_environment.yml` file.
To install, run from the `function_vectors/` directory:
```bash
conda env create -f fv_environment.yml
conda activate fv
```

The environment requires PyTorch ≥2.2.0 and `transformers>=4.51` to support all models listed below. PyTorch is installed via pip so the CUDA build is not pinned — install whichever `cu1xx` wheel matches your driver if the default does not work:
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

### HuggingFace access

Gemma-3 and Llama-2 are gated repositories that require accepting the license on HuggingFace before use. Once accepted, authenticate via one of:

**Option A — token file (recommended for cluster use):**
```bash
echo "hf_your_token_here" > ~/.hf_token
chmod 600 ~/.hf_token
```
The SLURM job scripts automatically load `~/.hf_token` into `HF_TOKEN` if the file exists.

**Option B — CLI login:**
```bash
huggingface-cli login
```

## Supported Models

The pipeline supports the following models out of the box:

| Model | HuggingFace ID | Notes |
|---|---|---|
| GPT-J 6B | `EleutherAI/gpt-j-6b` | Original paper model; universal heads hardcoded |
| GPT-2 XL | `gpt2-xl` | |
| Pythia family | `EleutherAI/pythia-*` | |
| Llama-2 7B/13B/70B | `meta-llama/Llama-2-*` | 70B uses 4-bit quantization; gated repo |
| Llama-3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | gated repo |
| Gemma-3 4B IT | `google/gemma-3-4b-it` | gated repo |
| Qwen3 8B | `Qwen/Qwen3-8B` | |
| OLMo | `allenai/OLMo-*` | |

## Demo Notebook
Checkout `notebooks/fv_demo.ipynb` for a jupyter notebook with a demo of how to create a function vector and use it in different contexts.

## Data
The datasets used in our project can be found in the `dataset_files` folder.

## Code
Our main evaluation scripts are contained in the `src` directory with sample script wrappers in `src/eval_scripts`.

Other main code is split into various util files:
- `eval_utils.py` contains code for evaluating function vectors in a variety of contexts
- `extract_utils.py`  contains functions for extracting function vectors and other relevant model activations.
- `intervention_utils.py` contains main functionality for intervening with function vectors during inference
- `model_utils.py` contains helpful functions for loading models & tokenizers from huggingface
- `prompt_utils.py` contains data loading and prompt creation functionality

## Running the Full Pipeline for New Models

For models whose universal head lists are not yet hardcoded in `extract_utils.py`, the pipeline has three phases.

### Phase 1 — Compute universal heads

Run `compute_universal_heads.py` for each new model. Following the paper, it aggregates indirect effects over all abstractive tasks for which 10-shot ICL top-1 accuracy exceeds the majority-label baseline. It prints a copy-pasteable `elif` block with the top-K heads:

```bash
python src/compute_universal_heads.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --save_path_root results/llama32_3b \
    --n_top_heads 15 \
    --n_shots 10 \
    --n_mean_trials 100 \
    --n_ie_trials 25 \
    --seed 42
```

All 29 tasks in `dataset_files/abstractive/` are candidates; the script prints how many passed the filter (e.g. `18/29 datasets`). Pass `--no_filter` to include all tasks regardless. Mean activations are saved for every task (needed for task-specific FV injection) even when a task fails the filter.

Copy the printed `elif` block into `src/utils/extract_utils.py` inside `compute_universal_function_vector` (before the `gpt-neox` entry, around line 442).

### Phase 2 — Layer sweep

Run `evaluate_function_vector.py` with `--edit_layer -1` to evaluate the function vector at every layer and find the best injection point:

```bash
python src/evaluate_function_vector.py \
    --dataset_name antonym \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --save_path_root results/llama32_3b \
    --n_top_heads 15 \
    --edit_layer -1 \
    --universal_set \
    --seed 42
```

Inspect `results/llama32_3b/antonym/zs_results_layer_sweep.json` to find the layer with the highest zero-shot accuracy, then use that layer in Phase 3.

### Phase 3 — Fixed-layer evaluation

```bash
python src/evaluate_function_vector.py \
    --dataset_name antonym \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --save_path_root results/llama32_3b \
    --n_top_heads 15 \
    --edit_layer 8 \
    --universal_set \
    --seed 42
```

## SLURM Batch Jobs

`src/eval_scripts/submit_hydra_jobs.py` generates and submits all three phases as SLURM jobs. The templates are straightforward to adapt to any SLURM cluster.

```bash
# Phase 1 — compute universal heads for new models (one job per model)
python src/eval_scripts/submit_hydra_jobs.py --phase compute_heads

# Phase 2 — layer sweep for all models × datasets (one job per pair)
python src/eval_scripts/submit_hydra_jobs.py --phase layer_sweep

# Phase 3 — fixed-layer eval using hardcoded L/3 defaults (no flag needed)
python src/eval_scripts/submit_hydra_jobs.py --phase fixed_eval

# Or override specific models after inspecting layer sweep results
python src/eval_scripts/submit_hydra_jobs.py --phase fixed_eval \
    --edit_layers gptj=9,llama32_3b=9,gemma3_4b=11,qwen3_8b=12

# n_heads sweep — find the optimal number of heads per model × dataset
python src/eval_scripts/submit_hydra_jobs.py --phase numheads_sweep \
    --edit_layers gemma3_4b=11 --models gemma3_4b

# Dry run — print sbatch commands without submitting
python src/eval_scripts/submit_hydra_jobs.py --phase layer_sweep --dry_run
```

Filter flags `--models` and `--datasets` accept comma-separated values and apply to all phases:
```bash
python src/eval_scripts/submit_hydra_jobs.py --phase layer_sweep \
    --models gemma3_4b --datasets antonym,english-french
```

The script uses the following per-model configuration (edit `MODELS` dict to add more):

| Model | Nick | Layers | Top-K heads | Default edit layer (L/3) |
|---|---|---|---|---|
| GPT-J 6B | `gptj` | 28 | 10 | 9 |
| Llama-3.2 3B Instruct | `llama32_3b` | 28 | 15 | 9 |
| Gemma-3 4B IT | `gemma3_4b` | 34 | 15 | 11 |
| Qwen3 8B | `qwen3_8b` | 36 | 25 | 12 |

SLURM script templates live in `src/eval_scripts/` (`hydra_compute_heads.sh`, `hydra_layer_sweep.sh`, `hydra_fixed_eval.sh`). The driver fills in model name, dataset, and layer placeholders and writes concrete `.sh` files to a timestamped subdirectory before submitting.

## Finding the Optimal Number of Heads

`src/test_numheads.py` sweeps `n_top_heads` from 1 to `--n_heads` and records zero-shot+FV accuracy at each step. It requires Phase 1 mean activations to already exist.

```bash
python src/test_numheads.py \
    --model_name google/gemma-3-4b-it \
    --model_nickname gemma3_4b \
    --dataset_name antonym \
    --save_path_root results/gemma3_4b \
    --edit_layer 11 \
    --n_heads 50
```

Results are saved to `results/<model_nick>/<model_nick>_test_numheads/<dataset>_perf_v_heads.json`. Submit as SLURM jobs across all datasets with:

```bash
python src/eval_scripts/submit_hydra_jobs.py --phase numheads_sweep \
    --edit_layers gemma3_4b=11 --models gemma3_4b
```

## Summarizing Results

All summary scripts are in `src/eval_scripts/` and should be run from `src/`.

**`summarize_results.py`** — main accuracy table across models and datasets after fixed-layer eval:
```bash
python eval_scripts/summarize_results.py --results_root ../results
```
Prints three tables (zero-shot+FV, shuffled-label+FV, ICL baseline) and saves to `results/summary.txt`.

**`summarize_layer_sweep.py`** — per-layer accuracy after the layer sweep, useful for picking the best injection layer:
```bash
# All models
python eval_scripts/summarize_layer_sweep.py --results_root ../results

# Single model or dataset
python eval_scripts/summarize_layer_sweep.py --results_root ../results --model gemma3_4b
python eval_scripts/summarize_layer_sweep.py --results_root ../results --model gemma3_4b --dataset antonym
```
Prints a layer × dataset accuracy table per model with the best layer highlighted, and saves to `results/layer_sweep_summary.txt`.

**`summarize_numheads.py`** — accuracy vs n_heads averaged across datasets, after running `test_numheads.py`:
```bash
python eval_scripts/summarize_numheads.py --results_root ../results --model_nick gemma3_4b

# Single dataset
python eval_scripts/summarize_numheads.py --results_root ../results --model_nick gemma3_4b --dataset antonym
```
Prints a n_heads × dataset accuracy table with the best n_heads highlighted, and saves to `results/<model_nick>_numheads_summary.txt`.

## Citing the original work
The paper by Todd et al. appeared at ICLR 2024 and can be cited as follows:

```bibtex
@inproceedings{todd2024function,
    title={Function Vectors in Large Language Models}, 
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    booktitle={The Twelfth International Conference on Learning Representations},
    url={https://openreview.net/forum?id=AwyxtyMwaG},
    note={arXiv:2310.15213},
    year={2024},
}
