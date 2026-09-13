"""Web UI for the tokenizer component.

Everything network/filesystem-facing (serving pages, reading the bundled
example texts, exposing the JSON API the page's JS calls) lives here, kept
separate from the actual tokenizer implementation in
``llm_from_scratch.data.tokenizer``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from llm_from_scratch.data.tokenizer import SimpleTokenizer
from viz.components.examples import EXAMPLES, read_example

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
    tokens = list(_tokenizer.tokenize([text]))
    return jsonify(tokens=tokens, count=len(tokens))


@bp.get("/api/examples/<example_id>")
def example_text(example_id: str):
    text = read_example(example_id)
    if text is None:
        return jsonify(error=f"unknown example {example_id!r}"), 404
    return jsonify(text=text)
