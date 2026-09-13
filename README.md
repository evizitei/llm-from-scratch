# llm-from-scratch

Working environment for going through *Build a Large Language Model (From Scratch)*.
Dependency management is via [uv](https://docs.astral.sh/uv/).

## Setup

Environment is already created (`.venv`, locked via `uv.lock`). To pick it back up:

```bash
uv sync
```

## Usage

Common commands are also available via [`just`](https://github.com/casey/just)
(`just` on its own lists them): `just sync`, `just test`, `just lint`,
`just fmt`, `just run`, `just lab`, `just viz`.

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

Run the test suite:

```bash
uv run pytest
```

Launch the component visualizer (a small Flask app for poking at pieces of
the codebase through a web UI — currently just the tokenizer):

```bash
uv run python -m viz.app
```

Then open http://127.0.0.1:5001/.

## Layout

- `src/llm_from_scratch/` — package code, loosely split by book topic:
  - `data/` — tokenization, dataset/dataloader construction
  - `attention/` — self-attention, causal attention, multi-head attention
  - `model/` — the GPT architecture itself (transformer blocks, layer norm, etc.)
  - `training/` — pretraining loop, loss, optimizer setup
  - `finetuning/` — classification and instruction finetuning
- `tests/` — pytest suite, mirroring the `src/llm_from_scratch/` layout
  (e.g. `tests/data/test_tokenizer.py` tests
  `src/llm_from_scratch/data/tokenizer.py`). Any downloading/caching of
  sample data for tests lives in `tests/conftest.py`, not in the library
  code itself.
- `viz/` — a small Flask app for interacting with pieces of the codebase in
  a browser. Each component (e.g. `viz/components/tokenizer/`) is a
  blueprint with its own routes, templates, static assets, and (where
  useful) bundled example text files.
- `notebooks/` — scratch/exploratory notebooks
- `data/` — downloaded datasets and model weights (gitignored, not committed)
