# Common project commands. Run `just` to list them.

default:
    just --list

# Sync the uv-managed environment (installs/updates deps from uv.lock).
# --all-groups is load-bearing: `test` and `viz` are non-default groups, so a
# plain `uv sync` would uninstall pytest and flask.
sync:
    uv sync --all-groups

# Run the test suite.
test:
    uv run pytest

# Lint with ruff.
lint:
    uv run ruff check .

# Format with ruff.
fmt:
    uv run ruff format .

# Run the main script.
run:
    uv run python src/llm_from_scratch/main.py

# Launch JupyterLab (kernel `llm-from-scratch` is pre-registered).
lab:
    uv run jupyter lab

# Launch the component visualizer at http://127.0.0.1:5001/.
viz:
    uv run python -m viz.app
