"""Bundled example texts shared by more than one component visualizer.

Kept in one place so the tokenizer and vocabulary-builder pages (and any
future component that wants some sample text to play with) all offer the
same examples instead of drifting apart.
"""

from __future__ import annotations

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent / "tokenizer" / "examples"

# (id, label, filename) for each bundled example text, in display order.
EXAMPLES = [
    ("the-verdict", "The Verdict (Edith Wharton, 1908)", "the-verdict.txt"),
    (
        "pride-and-prejudice",
        "Pride and Prejudice, ch. 1 (Jane Austen, 1813)",
        "pride-and-prejudice-ch1.txt",
    ),
    (
        "alice-in-wonderland",
        "Alice's Adventures in Wonderland, ch. 1 (Lewis Carroll, 1865)",
        "alice-in-wonderland-ch1.txt",
    ),
]
_EXAMPLE_FILENAMES = {example_id: filename for example_id, _, filename in EXAMPLES}


def read_example(example_id: str) -> str | None:
    """Return the text of the example named ``example_id``, or ``None``."""
    filename = _EXAMPLE_FILENAMES.get(example_id)
    if filename is None:
        return None
    return (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
