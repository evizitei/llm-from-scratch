# llm-from-scratch

Working environment for going through *Build a Large Language Model (From Scratch)*.
Dependency management is via [uv](https://docs.astral.sh/uv/).

## Setup

Environment is already created (`.venv`, locked via `uv.lock`). To pick it back up:

```bash
uv sync
```

## Usage

Run a script with the project's environment:

```bash
uv run python src/llm_from_scratch/main.py
```

Launch JupyterLab (kernel `llm-from-scratch` is pre-registered):

```bash
uv run jupyter lab
```

Add a new dependency:

```bash
uv add <package>
```

## Layout

- `src/llm_from_scratch/` — package code, loosely split by book topic:
  - `data/` — tokenization, dataset/dataloader construction
  - `attention/` — self-attention, causal attention, multi-head attention
  - `model/` — the GPT architecture itself (transformer blocks, layer norm, etc.)
  - `training/` — pretraining loop, loss, optimizer setup
  - `finetuning/` — classification and instruction finetuning
- `notebooks/` — scratch/exploratory notebooks
- `data/` — downloaded datasets and model weights (gitignored, not committed)
