"""Turning text into (context, next-token) training pairs.

This is the step between "we can turn text into token ids" and "we can
train something": a language model learns by being shown a run of tokens
and asked to predict the token that comes next, so the raw material for
training is a stream of ``(context, target)`` pairs.

The construction is the one from chapter 2 of *Build a Large Language
Model (From Scratch)*. Encode a chunk of text into ids, then line the
array up against itself shifted by one::

    ids:  [290, 4920, 2241, 287, 257]
    x:     290  4920  2241  287        (inputs)
    y:          4920  2241  287  257   (targets)

Reading that off one position at a time gives the pairs::

    [290]                  ----> 4920
    [290, 4920]            ----> 2241
    [290, 4920, 2241]      ----> 287
    [290, 4920, 2241, 287] ----> 257

Every pair is a free training example, and they all come from text that
nobody had to label -- which is the whole trick behind training on the
open internet.

Windows and stride
------------------

Growing the context forever is the illustration, not the mechanism: a
model has a fixed context length, and a causal transformer computes every
one of those growing-prefix predictions *at once*. A window of ``L``
tokens masked so position ``i`` only sees ``0..i`` is exactly the ``L``
pairs above, in a single forward pass. So the shape training actually
wants is ``window()``: a fixed-length ``x`` and the same run shifted by
one as ``y``.

``max_length`` is how wide that window is. ``stride`` is how far it slides
before the next one is taken, and the two together decide how much the
windows overlap::

    ids: [0 1 2 3 4 5 6 7 8 9], max_length=4

    stride=4 (no overlap)      stride=1 (maximal overlap)
    x=[0,1,2,3] y=[1,2,3,4]    x=[0,1,2,3] y=[1,2,3,4]
    x=[4,5,6,7] y=[5,6,7,8]    x=[1,2,3,4] y=[2,3,4,5]
                               x=[2,3,4,5] y=[3,4,5,6]
                               ...

Overlap is tempting -- more windows out of the same text -- but it is
mostly an illusion. Because one window already trains on every position
inside it, an overlapping window re-trains on prediction tasks the
previous one covered, just shifted along. That is duplicated data (which
pushes a model toward memorizing) at several times the compute. Hence the
default here, and the book's choice: ``stride = max_length``, tiling the
text edge to edge so every token is predicted exactly once.

The honest argument *for* overlap is position diversity. With no overlap,
a given token is only ever predicted from one amount of context -- the
token that lands at the start of a window always has none. Overlap lets
the same text be learned at several context lengths, which is worth the
duplication on a corpus small enough that data, not compute, is the
constraint. A stride *larger* than ``max_length`` goes the other way and
skips text, a way to subsample a corpus too big to use in full.

One trap worth knowing: overlapping windows leak across a train/validation
split, because a validation window can share tokens with a training one.
Split the text first, then window each side.

Chunk boundaries matter
-----------------------

``DataSampler`` takes a *stream* of text chunks (a list of documents, a
generator reading files one at a time -- the same shape
``SimpleTokenizer`` and ``build_vocabulary`` accept) and starts over at
each chunk. Context never runs across a boundary: the last words of one
document are not evidence about the first words of the next, and letting
them leak in would teach the model a relationship that isn't there. If
two chunks really are continuous, hand them over as one chunk.

Encoders
--------

``DataSampler`` only needs something that can turn a string into ids and
ids back into a string -- the ``TextEncoder`` protocol below. That shape
is deliberately tiktoken's, so ``tiktoken.get_encoding("gpt2")`` can be
passed straight in. To use this project's own word-level vocabulary
instead, wrap a ``VocabularyTokenizer`` in ``VocabularyEncoder``, which
adapts its streaming interface to the same two methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple, Protocol

from llm_from_scratch.data.tokenizer import VocabularyTokenizer


class TextEncoder(Protocol):
    """Anything that can turn text into token ids and back.

    Deliberately the subset of ``tiktoken.Encoding``'s interface that this
    module needs, so a real BPE encoder satisfies it as-is::

        sampler = DataSampler(tiktoken.get_encoding("gpt2"))
    """

    def encode(self, text: str) -> Sequence[int]:
        """Return the token ids for ``text``."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Return the text for a sequence of token ids."""
        ...


class VocabularyEncoder:
    """Adapts a ``VocabularyTokenizer`` to the ``TextEncoder`` protocol.

    ``VocabularyTokenizer`` streams: it takes an iterable of text chunks
    and yields ids one at a time, which is what you want for feeding a
    corpus through a vocabulary without holding it in memory.
    ``TextEncoder`` is the opposite shape -- one string in, a whole list of
    ids out -- because that is the shape tiktoken has, and matching it is
    what lets a toy vocabulary and a real BPE encoder be used
    interchangeably here.

    A pair's context has to be materialized anyway, so nothing is lost by
    collecting one chunk at a time.
    """

    def __init__(self, tokenizer: VocabularyTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> VocabularyTokenizer:
        """The tokenizer this encoder wraps."""
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        """Return the token ids for ``text``."""
        return list(self._tokenizer.encode([text]))

    def decode(self, ids: Sequence[int]) -> str:
        """Return the (reconstructed) text for a sequence of token ids."""
        return "".join(self._tokenizer.decode(ids))


class ContextPair(NamedTuple):
    """One training example: the tokens so far, and the one that follows.

    A ``NamedTuple`` so it unpacks like the plain tuple it conceptually
    is::

        for context, target in sampler.sample(text):
            ...
    """

    #: The token ids seen so far, oldest first. Never empty.
    context: tuple[int, ...]

    #: The id of the token the model is supposed to predict next.
    target: int


class TokenWindow(NamedTuple):
    """One fixed-length training window: inputs, and the same run shifted.

    ``targets[i]`` is the token that follows ``inputs[i]``, so a causal
    model trained on this window learns every next-token prediction inside
    it at once. Both are ``max_length`` long.

    A ``NamedTuple``, so it unpacks::

        for inputs, targets in sampler.window(text):
            ...
    """

    #: ``max_length`` token ids, the model's input.
    inputs: tuple[int, ...]

    #: The same run advanced by one position: what to predict at each step.
    targets: tuple[int, ...]


class DataSampler:
    """Cuts a stream of text into next-token training examples.

    Two shapes come out of the same text, and which you want depends on
    what you are doing with it. ``window`` yields the fixed-length
    ``(inputs, targets)`` arrays a causal model is actually trained on.
    ``sample`` yields the one-pair-at-a-time growing contexts that show
    what a window *means*, which is the better thing to put in front of a
    person.

    Parameters
    ----------
    encoder:
        How text becomes token ids -- a ``tiktoken`` encoding, a
        ``VocabularyEncoder`` around this project's own vocabulary, or
        anything else matching ``TextEncoder``.
    max_length:
        The context length: how many tokens a window holds, and the most
        context any single pair from ``sample`` will carry. ``None``
        (the default) leaves ``sample`` unbounded and makes ``window``
        unusable -- there is no window without a width.
    stride:
        How far ``window`` advances between windows. Defaults to
        ``max_length``, which tiles the text with no overlap and no gaps;
        see the module docstring for when to want something else. Ignored
        by ``sample``.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        max_length: int | None = None,
        stride: int | None = None,
    ) -> None:
        if max_length is not None and max_length < 1:
            raise ValueError(f"max_length must be at least 1, got {max_length}")
        if stride is not None and stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        self._encoder = encoder
        self._max_length = max_length
        self._stride = stride if stride is not None else max_length

    @property
    def encoder(self) -> TextEncoder:
        """The encoder this sampler encodes with."""
        return self._encoder

    @property
    def max_length(self) -> int | None:
        """The context length, or ``None`` if contexts are unbounded."""
        return self._max_length

    @property
    def stride(self) -> int | None:
        """How far ``window`` advances between windows."""
        return self._stride

    def sample(self, texts: str | Iterable[str]) -> Iterator[ContextPair]:
        """Yield every ``(context, target)`` pair in ``texts``, in order.

        ``texts`` is either a single string or an iterable of them, pulled
        and encoded one chunk at a time. Each chunk yields one pair per
        token after the first: a context of everything up to that point,
        and that token as the target. Contexts reset at every chunk
        boundary, so no pair ever straddles two chunks -- see the module
        docstring for why.

        If this sampler has a ``max_length``, contexts stop growing there
        and start sliding instead: each one keeps the most recent
        ``max_length`` tokens and drops what fell off the front, which is
        what a fixed context length does to a model's view of the text.

        A chunk that encodes to fewer than two ids yields nothing; there
        is no pair to make from a single token with nothing following it.
        """
        max_length = self._max_length
        for ids in self._encode_chunks(texts):
            for index in range(1, len(ids)):
                start = 0 if max_length is None else max(0, index - max_length)
                yield ContextPair(context=ids[start:index], target=ids[index])

    def window(self, texts: str | Iterable[str]) -> Iterator[TokenWindow]:
        """Yield fixed-length ``(inputs, targets)`` windows over ``texts``.

        Each window is ``max_length`` ids and the same run shifted one
        position, advancing by ``stride`` between windows. This is the
        form a causal model trains on; ``sample`` is the same information
        unrolled for a human to read.

        Windows never straddle a chunk boundary, and a chunk shorter than
        ``max_length + 1`` ids yields nothing -- a partial window has no
        well-defined shape to hand a model, so the tail of a chunk that
        does not divide evenly is dropped rather than padded.

        Raises ``ValueError`` if this sampler was built without a
        ``max_length``.
        """
        max_length = self._max_length
        if max_length is None:
            raise ValueError(
                "window() needs a max_length; construct the sampler with one"
            )
        stride = self._stride
        assert stride is not None  # set alongside max_length
        for ids in self._encode_chunks(texts):
            # Stop where a full window plus its shifted target still fits.
            for start in range(0, len(ids) - max_length, stride):
                end = start + max_length
                yield TokenWindow(
                    inputs=ids[start:end], targets=ids[start + 1 : end + 1]
                )

    def _encode_chunks(self, texts: str | Iterable[str]) -> Iterator[tuple[int, ...]]:
        """Encode each chunk of ``texts`` in turn, one chunk at a time.

        A bare string is one chunk, not a stream of single characters.
        Each chunk's ids are materialized so windows and contexts can be
        sliced out of them, but only ever one chunk's worth at a time.
        """
        for text in [texts] if isinstance(texts, str) else texts:
            yield tuple(self._encoder.encode(text))

    def decode_pair(self, pair: ContextPair) -> tuple[str, str]:
        """Render a pair as ``(context text, target text)``, for humans.

        Handy for eyeballing what a sampler is actually producing --
        ``("I HAD always thought", " Jack")`` reads a great deal better
        than two lists of integers. Decoding is lossy for some encoders
        (see ``VocabularyTokenizer.decode``), so this is for inspection,
        not for round-tripping.
        """
        return (
            self._encoder.decode(pair.context),
            self._encoder.decode([pair.target]),
        )

    def decode_window(self, window: TokenWindow) -> tuple[str, str]:
        """Render a window as ``(inputs text, targets text)``, for humans.

        The same inspection aid as ``decode_pair``, and lossy in the same
        way. Seeing the two strings side by side is the clearest way to
        watch the one-position shift that makes the targets targets.
        """
        return (
            self._encoder.decode(window.inputs),
            self._encoder.decode(window.targets),
        )
