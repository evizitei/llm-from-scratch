"""Shared pytest fixtures.

Anything that reaches out to the network or the filesystem for test data
lives here, deliberately kept separate from the library code under
``src/llm_from_scratch``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

THE_VERDICT_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)

# Cached alongside the project's other downloaded data (gitignored).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
THE_VERDICT_PATH = DATA_DIR / "the-verdict.txt"


@pytest.fixture(scope="session")
def the_verdict_text() -> str:
    """Return the text of "The Verdict", downloading and caching it once.

    Skips the test (rather than failing) if the file isn't already cached
    and there's no network access to fetch it.
    """
    if not THE_VERDICT_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(THE_VERDICT_URL, THE_VERDICT_PATH)
        except OSError as exc:
            pytest.skip(f"could not download the-verdict.txt: {exc}")

    return THE_VERDICT_PATH.read_text(encoding="utf-8")
