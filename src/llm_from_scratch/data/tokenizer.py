"""A minimal, regex-based tokenizer, plus a directory-of-files text source.

This mirrors the "toy" tokenizer built up in chapter 2 of *Build a Large
Language Model (From Scratch)*: text is split on whitespace and a fixed set
of punctuation characters, and the resulting whitespace-only fragments are
dropped.

``SimpleTokenizer`` works iterator-to-iterator: it takes an iterator over
chunks of text (a list held in memory, a generator reading files lazily, or
anything else) and returns an iterator over tokens. It pulls one chunk at a
time, yields that chunk's tokens one at a time, and only then pulls the
next chunk -- so it never needs the whole input, or the whole token stream,
in memory at once. That makes it equally happy tokenizing a short string
(wrapped in a one-element list) or an arbitrarily large body of text spread
across many files, e.g. via ``iter_directory_texts``.

This module deliberately knows nothing about vocabularies -- see
``llm_from_scratch.data.vocabulary`` for that. The two are independent: a
token stream produced here is just an iterator over strings, and anything
that consumes one (``build_vocabulary`` or otherwise) needs nothing more
than that.

This module also contains only tokenization logic itself. Any code that
downloads sample text, prints results, or otherwise deals with the outside
world belongs elsewhere (see ``tests/data/test_tokenizer.py`` and
``viz/components/tokenizer/`` for that).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

# Splits on: whitespace, common punctuation ,.:;?_!"()' (including curly
# "smart" quotes “”‘’, as seen in e.g. Pride and Prejudice) and em/en-dashes
# rendered as double hyphens ("--"). Capturing groups in re.split keep the
# delimiters themselves as tokens in the result.
_DEFAULT_PATTERN = re.compile(r'([,.:;?_!"()\'“”‘’]|--|\s)')


class SimpleTokenizer:
    """Splits a stream of text chunks into a stream of tokens using a regex.

    Parameters
    ----------
    pattern:
        A compiled regex with at least one capturing group, used with
        ``re.split`` to break text apart. Defaults to the punctuation/
        whitespace pattern used in the book. The pattern's capturing
        group(s) determine which delimiters are kept as tokens.
    """

    def __init__(self, pattern: re.Pattern[str] = _DEFAULT_PATTERN) -> None:
        self._pattern = pattern

    def tokenize(self, texts: Iterable[str]) -> Iterator[str]:
        """Tokenize a stream of text chunks into a stream of tokens.

        ``texts`` is consumed one chunk at a time; each chunk is split into
        tokens and yielded before the next chunk is pulled, so this never
        holds more than one chunk's worth of text (and tokens) in memory at
        once. Whitespace-only fragments are dropped, and every remaining
        token is stripped of leading/trailing whitespace.
        """
        for text in texts:
            pieces = self._pattern.split(text)
            yield from (piece.strip() for piece in pieces if piece.strip())


def iter_directory_texts(directory: str | Path) -> Iterator[str]:
    """Yield the contents of each file in ``directory``, one file at a time.

    Files are visited in name order (for repeatable results) and read as
    whole strings; subdirectories are not descended into. This function
    knows nothing about tokenization -- it only knows how to fetch text out
    of a directory of files, so it can be swapped for a different text
    source (an in-memory list, a database, ...) without touching how that
    text is later tokenized or turned into a vocabulary.
    """
    directory = Path(directory)
    for path in sorted(directory.iterdir()):
        if path.is_file():
            yield path.read_text(encoding="utf-8")
