"""Web UI for the data sampler component.

Everything network/filesystem-facing (serving pages, reading the bundled
example texts, loading the GPT-2 BPE ranks, exposing the JSON API the
page's JS calls) lives here, kept separate from the actual sampling logic
in ``llm_from_scratch.data.sampler``.

The page offers both encoders the sampler supports, because the pairs look
quite different through each: GPT-2's BPE splits unfamiliar words into
sub-word pieces and keeps leading spaces, while the toy word-level
vocabulary has one id per whole word and throws the spacing away.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from llm_from_scratch.data.sampler import (
    ContextPair,
    DataSampler,
    TokenWindow,
    VocabularyEncoder,
)
from llm_from_scratch.data.tokenizer import SimpleTokenizer, VocabularyTokenizer
from llm_from_scratch.data.vocabulary import build_vocabulary
from viz.components.examples import EXAMPLES, read_example

bp = Blueprint(
    "sampler",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/sampler",
    url_prefix="/sampler",
)

component = {
    "slug": "sampler",
    "name": "Data Sampler",
    "description": (
        "Cut text into the (input, target) windows a model trains on, and "
        "watch context size and stride change what comes out."
    ),
    "blueprint": bp,
}

# A page full of windows is for reading, not for training -- past a couple
# of hundred rows nobody is looking, and the JSON gets big enough to make
# the page sluggish. The response says how many were cut off.
MAX_ROWS = 200

_tokenizer = SimpleTokenizer()

#: Lazily loaded: ``tiktoken.get_encoding`` downloads the BPE ranks the
#: first time it runs, so it shouldn't happen at import (and shouldn't
#: happen twice).
_gpt2_encoder = None


def _load_gpt2_encoder():
    """Return the GPT-2 BPE encoding, loading it on first use."""
    global _gpt2_encoder
    if _gpt2_encoder is None:
        import tiktoken

        _gpt2_encoder = tiktoken.get_encoding("gpt2")
    return _gpt2_encoder


def _toy_encoder(text: str) -> VocabularyEncoder:
    """A word-level encoder over a vocabulary built from ``text`` itself.

    Building the vocabulary from the very text being sampled is what keeps
    the toy encoder legible here: every word in the box gets a real id
    rather than collapsing into ``<|unk|>``.
    """
    vocabulary = build_vocabulary(_tokenizer.tokenize([text]))
    return VocabularyEncoder(VocabularyTokenizer(vocabulary))


def _build_encoder(name: str, text: str):
    """Return the encoder named ``name``, or raise ``ValueError``."""
    if name == "toy":
        return _toy_encoder(text)
    if name == "gpt2":
        return _load_gpt2_encoder()
    raise ValueError(f"unknown encoder {name!r}")


def _positive_int(payload: dict, key: str, default: int) -> int:
    """Read a positive integer field, falling back to ``default``."""
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def _tokens(encoder, ids) -> list[dict]:
    """Pair each id with its own decoded text, for side-by-side display."""
    return [{"id": id_, "text": encoder.decode([id_])} for id_ in ids]


def _window_row(sampler: DataSampler, start: int, window: TokenWindow) -> dict:
    encoder = sampler.encoder
    inputs_text, targets_text = sampler.decode_window(window)
    return {
        "start": start,
        "inputs": _tokens(encoder, window.inputs),
        "targets": _tokens(encoder, window.targets),
        "inputs_text": inputs_text,
        "targets_text": targets_text,
    }


def _pair_row(sampler: DataSampler, pair: ContextPair) -> dict:
    encoder = sampler.encoder
    context_text, target_text = sampler.decode_pair(pair)
    return {
        "context": _tokens(encoder, pair.context),
        "target": {"id": pair.target, "text": encoder.decode([pair.target])},
        "context_text": context_text,
        "target_text": target_text,
    }


@bp.get("/")
def index():
    examples = [{"id": example_id, "label": label} for example_id, label, _ in EXAMPLES]
    return render_template("sampler.html", examples=examples)


@bp.post("/api/sample")
def sample():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    max_length = _positive_int(payload, "max_length", 4)
    stride = _positive_int(payload, "stride", max_length)
    view = payload.get("view", "windows")

    try:
        encoder = _build_encoder(payload.get("encoder", "gpt2"), text)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001
        # tiktoken fetches the BPE ranks over the network on first use and
        # raises whatever its HTTP client raises. Which exception that is
        # isn't worth pinning down here: any failure to load the encoder is
        # the same answer to the page, and a 503 with the reason beats a
        # stack trace in the dev server log.
        return jsonify(error=f"could not load the GPT-2 encoder: {exc}"), 503

    sampler = DataSampler(encoder, max_length=max_length, stride=stride)
    token_count = len(encoder.encode(text))

    # Pull one row past the cap so the page can say whether it truncated
    # without counting the whole (possibly enormous) stream.
    if view == "pairs":
        pairs = sampler.sample(text)
        rows = [_pair_row(sampler, pair) for pair, _ in zip(pairs, range(MAX_ROWS + 1))]
    else:
        windows = enumerate(sampler.window(text))
        rows = [
            _window_row(sampler, index * stride, window)
            for (index, window), _ in zip(windows, range(MAX_ROWS + 1))
        ]

    truncated = len(rows) > MAX_ROWS
    return jsonify(
        view=view,
        rows=rows[:MAX_ROWS],
        truncated=truncated,
        token_count=token_count,
        max_length=max_length,
        stride=stride,
    )


@bp.get("/api/examples/<example_id>")
def example_text(example_id: str):
    text = read_example(example_id)
    if text is None:
        return jsonify(error=f"unknown example {example_id!r}"), 404
    return jsonify(text=text)
