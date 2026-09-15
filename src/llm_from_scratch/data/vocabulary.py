"""Building a token -> integer vocabulary from a stream of tokens.

This module knows nothing about tokenizers, files, or where tokens come
from -- it only knows how to consume an iterator over strings, one token at
a time, and turn that into a ``Vocabulary``. Anything that produces such an
iterator (``llm_from_scratch.data.tokenizer.SimpleTokenizer.tokenize``, a
plain list held in memory, or anything else) can be plugged in without this
module needing to know about it.

A ``Vocabulary`` is a *lookup table*, and nothing more: it answers what id a
token has, what token an id has, and which of its entries are reserved. It
does not run streams of text or ids through itself -- that is
``llm_from_scratch.data.tokenizer.VocabularyTokenizer``'s job, and the
dependency points that way only. A vocabulary never needs a tokenizer.

Out-of-vocabulary tokens
------------------------

Any word-level vocabulary built from a finite corpus will meet words it has
never seen. The standard answer is a *special token*: a reserved entry that
stands for something other than a literal word from the corpus. This module
reserves two by default, following the convention this book's GPT models
use:

* ``<|unk|>`` -- substituted for any token missing from the vocabulary, so
  encoding never fails on unseen input.
* ``<|endoftext|>`` -- a document separator, so unrelated texts concatenated
  into one training stream don't bleed into each other.

The ``<|...|>`` delimiters are deliberately ugly: the point is a string that
will never show up in natural text and be confused for a real token. (BERT
and friends use ``[UNK]``/``[PAD]``/``[CLS]`` for the same reason.)

Special tokens are assigned the *lowest* ids, before any corpus token. That
keeps their ids stable when the corpus changes -- which matters, because an
id is a row index into a model's embedding matrix, and a trained checkpoint
is only meaningful with the exact vocabulary it was trained against.

The broader industry answer, worth knowing while writing this one: modern
tokenizers mostly make ``<|unk|>`` unnecessary. Byte-level BPE (GPT-2
onwards) has every byte in its base vocabulary, so any input at all decomposes
into known pieces and nothing is ever out of vocabulary. WordPiece
(BERT) and SentencePiece fall back to characters similarly, and only need
an unknown token for the rare byte they can't represent. This word-level
vocabulary is the pedagogical stepping stone to that, not the destination.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Stands in for any token that isn't in the vocabulary.
UNKNOWN_TOKEN = "<|unk|>"

#: Marks a boundary between unrelated documents in a single token stream.
END_OF_TEXT_TOKEN = "<|endoftext|>"

#: Reserved by ``build_vocabulary`` unless you ask for something else.
#: Order matters: these are assigned ids 0, 1, ... in this order.
DEFAULT_SPECIAL_TOKENS = (UNKNOWN_TOKEN, END_OF_TEXT_TOKEN)


class Vocabulary:
    """A light, dict-like wrapper around a token -> integer id mapping.

    This does nothing clever -- it answers the handful of questions that
    come up once you have a token/id mapping (how big is it, is this token
    in it, what id does it map to) and streams tokens through it in either
    direction.

    Parameters
    ----------
    token_to_id:
        The mapping itself. Ids are expected to be distinct.
    special_tokens:
        Which of those tokens are reserved rather than drawn from the
        corpus. Recorded so ``decode`` can filter them out and so
        ``encode`` knows what to substitute for unknown tokens; this
        constructor does not add them to ``token_to_id`` for you. Any
        listed token that isn't in the mapping is ignored.
    """

    def __init__(
        self,
        token_to_id: dict[str, int],
        special_tokens: Iterable[str] = (),
    ) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = {id_: token for token, id_ in token_to_id.items()}
        self._special_tokens = tuple(t for t in special_tokens if t in token_to_id)
        self._special_ids = frozenset(token_to_id[t] for t in self._special_tokens)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __getitem__(self, token: str) -> int:
        return self._token_to_id[token]

    @property
    def special_tokens(self) -> tuple[str, ...]:
        """The reserved tokens in this vocabulary, in id order."""
        return self._special_tokens

    @property
    def unknown_id(self) -> int | None:
        """The id of ``UNKNOWN_TOKEN``, or ``None`` if it isn't reserved."""
        return self._token_to_id.get(UNKNOWN_TOKEN)

    def is_special_id(self, id_: int) -> bool:
        """Whether ``id_`` belongs to a reserved token rather than a word."""
        return id_ in self._special_ids

    def token_to_id(self, token: str) -> int:
        """Return the integer id for ``token``.

        Raises ``KeyError`` for an unknown token. This is the strict
        lookup -- substituting ``<|unk|>`` for tokens that aren't here is
        the tokenizer's decision to make, not the table's.
        """
        return self._token_to_id[token]

    def id_to_token(self, id_: int) -> str:
        """Return the token for integer id ``id_``."""
        return self._id_to_token[id_]


def build_vocabulary(
    tokens: Iterable[str],
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
) -> Vocabulary:
    """Build a ``Vocabulary`` from a stream of tokens.

    ``special_tokens`` are assigned ids first, in the order given, and the
    corpus tokens follow in alphabetical order. Pass an empty sequence for
    a vocabulary of nothing but corpus tokens -- at the cost of leaving a
    tokenizer with nothing to substitute for an unseen word.

    ``tokens`` is consumed one token at a time, so it works equally well
    whether the tokens are already in memory or are being produced lazily
    (e.g. from files read one at a time). This function only ever deals in
    strings -- it has no idea where they came from or how they were
    tokenized.
    """
    special_tokens = tuple(dict.fromkeys(special_tokens))

    unique_tokens: set[str] = set()
    for token in tokens:
        unique_tokens.add(token)
    unique_tokens.difference_update(special_tokens)

    ordered = list(special_tokens) + sorted(unique_tokens)
    token_to_id = {token: index for index, token in enumerate(ordered)}
    return Vocabulary(token_to_id, special_tokens=special_tokens)
