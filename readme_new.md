# PoisonedRAG API-Only Setup

This guide documents the minimal setup for running PoisonedRAG with an API model through OpenRouter. It avoids local LLM serving dependencies such as `fschat`, does not require a local Vicuna/LLaMA model, and does not require Torch for the API-only path.

## Environment

Use Python 3.10. A conda environment is recommended.

```bash
conda create -n PoisonedRAG python=3.10 pip
conda activate PoisonedRAG
```

Install only the libraries needed for the OpenRouter API workflow and BEIR dataset loading.

```bash
pip install beir openai tqdm numpy charset-normalizer
```

Notes:

- `openai` is used for both OpenAI-compatible APIs and OpenRouter.
- `beir` downloads and loads the base datasets.
- `torch`, `transformers`, and `sentence-transformers` are not required for the API-only OpenRouter run when `attack_method` is `None`.
- `torch`, `transformers`, and `sentence-transformers` are only needed for retriever/attack modes that load local embedding models.
- Local LLM packages such as `fschat[model_worker,webui]` are not required for the OpenRouter API setup.
- The dataset downloader in `src/utils.py` and `prepare_dataset.py` uses Python standard-library download/unzip logic, avoiding BEIR's Torch-dependent `util` import.

In this workspace, a temporary conda setup was created at:

```text
/tmp/miniforge-poisonedrag
/tmp/poisonedrag-api
```

Activate it with:

```bash
source /tmp/miniforge-poisonedrag/bin/activate /tmp/poisonedrag-api
```

## OpenRouter API Setup

The default API config is:

```text
model_configs/openrouter_gpt4_config.json
```

It uses:

```json
{
  "provider": "gpt",
  "name": "~openai/gpt-latest",
  "base_url": "https://openrouter.ai/api/v1"
}
```

The API key is read from the `OPENROUTER_API_KEY` environment variable.

Create a local `.env` file in the project root:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The GPT wrapper loads `.env` automatically, so exporting the variable manually is optional. If you prefer shell exports:

```bash
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Dataset Setup

The base dataset setting is `nq` with the `test` split.

Current defaults in `run.py`:

```python
test_params = {
    "eval_model_code": "contriever",
    "eval_dataset": "nq",
    "split": "test",
    "query_results_dir": "main",
    "model_name": "openrouter_gpt4",
    "use_truth": False,
    "top_k": 5,
    "gpu_id": 0,
    "attack_method": None,
    "adv_per_query": 5,
    "score_function": "dot",
    "repeat_times": 1,
    "M": 1,
    "seed": 12,
    "note": None,
}
```

Supported BEIR datasets in this repo:

- `nq`
- `hotpotqa`
- `msmarco`

Datasets are downloaded automatically into `datasets/` when needed. To download them manually:

```bash
python prepare_dataset.py
```

## Minimal API Run

From the project root:

```bash
python main.py \
  --eval_model_code contriever \
  --eval_dataset nq \
  --split test \
  --query_results_dir main \
  --model_name openrouter_gpt4 \
  --top_k 5 \
  --use_truth False \
  --attack_method None \
  --repeat_times 1 \
  --M 1 \
  --name nq-openrouter-api-smoke
```

This uses `model_configs/openrouter_gpt4_config.json` because `model_name` is set to:

```python
"model_name": "openrouter_gpt4"
```

Logs are written under:

```text
logs/main_logs/
```

Query results are written under:

```text
results/query_results/main/
```

Verified smoke-test output in this workspace:

```text
results/query_results/main/nq-openrouter-api-smoke.json
```

The verified run used:

```text
dataset: nq
split: test
model: openrouter_gpt4
attack_method: None
repeat_times: 1
M: 1
```

## Minimal Smoke Test

For a smaller API-cost run, temporarily reduce the workflow size in `run.py`:

```python
"repeat_times": 1,
"M": 1,
```

This runs one target query while keeping the same dataset, retrieval settings, prompt path, and OpenRouter API model.

## Important Runtime Notes

- The API-only default in this workspace is `attack_method=None`.
- `LM_targeted` and `hotflip` load retriever/attack dependencies and are not part of the minimal OpenRouter setup.
- For an API-only answer-generation check without adversarial retrieval injection, keep:

```python
"attack_method": None,
```

- `main.py` has been adjusted so CUDA/Torch are only loaded when attack mode is enabled.
- Existing retrieval result files are already present under `results/beir_results/`, but the BEIR corpus files are still needed in `datasets/` so the code can load document text.

## Remote Repository

Current git remotes:

```text
origin   https://github.com/Mubeen-Fatima/PoisonedRAG.git
upstream https://github.com/sleeepeer/PoisonedRAG.git
```
