"""Building a token -> integer vocabulary from a stream of tokens.

This module knows nothing about tokenizers, files, or where tokens come
from -- it only knows how to consume an iterator over strings, one token at
a time, and turn that into a ``Vocabulary``. Anything that produces such an
iterator (``llm_from_scratch.data.tokenizer.iter_tokens``, a plain list held
in memory, or anything else) can be plugged in without either module needing
to import the other.
"""

from __future__ import annotations

from collections.abc import Iterable


class Vocabulary:
    """A light, dict-like wrapper around a token -> integer id mapping.

    This does nothing clever -- it just answers the handful of questions
    that come up once you have a token/id mapping: how many tokens are in
    it, whether a given token is one of them, and what id (or token) a
    token (or id) maps to.
    """

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = {id_: token for token, id_ in token_to_id.items()}

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __getitem__(self, token: str) -> int:
        return self._token_to_id[token]

    def token_to_id(self, token: str) -> int:
        """Return the integer id for ``token``."""
        return self._token_to_id[token]

    def id_to_token(self, id_: int) -> str:
        """Return the token for integer id ``id_``."""
        return self._id_to_token[id_]


def build_vocabulary(tokens: Iterable[str]) -> Vocabulary:
    """Build an alphabetically sorted ``Vocabulary`` from a stream of tokens.

    ``tokens`` is consumed one token at a time, so it works equally well
    whether the tokens are already in memory or are being produced lazily
    (e.g. from files read one at a time). This function only ever deals in
    strings -- it has no idea where they came from or how they were
    tokenized.
    """
    unique_tokens: set[str] = set()
    for token in tokens:
        unique_tokens.add(token)

    token_to_id = {token: index for index, token in enumerate(sorted(unique_tokens))}
    return Vocabulary(token_to_id)
