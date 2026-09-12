"""Web UI for the tokenizer component.

Everything network/filesystem-facing (serving pages, reading the bundled
example texts, exposing the JSON API the page's JS calls) lives here, kept
separate from the actual tokenizer implementation in
``llm_from_scratch.data.tokenizer``.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from llm_from_scratch.data.tokenizer import SimpleTokenizer

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

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

bp = Blueprint(
    "tokenizer",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/tokenizer",
    url_prefix="/tokenizer",
)

_tokenizer = SimpleTokenizer()

component = {
    "slug": "tokenizer",
    "name": "Tokenizer",
    "description": (
        "Split text into word and punctuation tokens with the regex-based "
        "toy tokenizer."
    ),
    "blueprint": bp,
}


@bp.get("/")
def index():
    examples = [{"id": example_id, "label": label} for example_id, label, _ in EXAMPLES]
    return render_template("tokenizer.html", examples=examples)


@bp.post("/api/tokenize")
def tokenize():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    tokens = _tokenizer.tokenize(text)
    return jsonify(tokens=tokens, count=len(tokens))


@bp.get("/api/examples/<example_id>")
def example_text(example_id: str):
    filename = _EXAMPLE_FILENAMES.get(example_id)
    if filename is None:
        return jsonify(error=f"unknown example {example_id!r}"), 404
    text = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
    return jsonify(text=text)
