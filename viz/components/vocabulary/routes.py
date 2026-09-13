"""Web UI for the vocabulary-builder component.

Everything network/filesystem-facing (serving pages, reading the bundled
example texts, exposing the JSON API the page's JS calls) lives here, kept
separate from the actual vocabulary-building logic in
``llm_from_scratch.data.tokenizer``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from llm_from_scratch.data.tokenizer import SimpleTokenizer
from llm_from_scratch.data.vocabulary import build_vocabulary
from viz.components.examples import EXAMPLES, read_example

bp = Blueprint(
    "vocabulary",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/vocabulary",
    url_prefix="/vocabulary",
)

component = {
    "slug": "vocabulary",
    "name": "Vocabulary Builder",
    "description": (
        "Build an alphabetically sorted token → integer vocabulary "
        "from typed text or bundled example texts."
    ),
    "blueprint": bp,
}

_tokenizer = SimpleTokenizer()


@bp.get("/")
def index():
    examples = [{"id": example_id, "label": label} for example_id, label, _ in EXAMPLES]
    return render_template("vocabulary.html", examples=examples)


@bp.post("/api/build")
def build():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    vocabulary = build_vocabulary(_tokenizer.tokenize([text]))
    entries = [{"token": vocabulary.id_to_token(i), "id": i} for i in range(len(vocabulary))]
    return jsonify(vocabulary=entries, size=len(vocabulary))


@bp.get("/api/examples/<example_id>")
def example_text(example_id: str):
    text = read_example(example_id)
    if text is None:
        return jsonify(error=f"unknown example {example_id!r}"), 404
    return jsonify(text=text)
