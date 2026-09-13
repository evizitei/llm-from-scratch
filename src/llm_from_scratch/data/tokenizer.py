"""A minimal, regex-based tokenizer, plus helpers for streaming text into it.

This mirrors the "toy" tokenizer built up in chapter 2 of *Build a Large
Language Model (From Scratch)*: text is split on whitespace and a fixed set
of punctuation characters, and the resulting whitespace-only fragments are
dropped.

- ``SimpleTokenizer`` turns one string of text into a list of tokens.
- ``iter_tokens`` / ``iter_directory_texts`` turn *some source* of text
  (an in-memory iterable, a directory of files, ...) into a flat stream of
  tokens. Adding a new source only ever means writing a new "iterator over
  text" function -- nothing downstream needs to change.

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

# Splits on: whitespace, common punctuation ,.:;?_!"()' and em/en-dashes
# rendered as double hyphens ("--"). Capturing groups in re.split keep the
# delimiters themselves as tokens in the result.
_DEFAULT_PATTERN = re.compile(r'([,.:;?_!"()\']|--|\s)')


class SimpleTokenizer:
    """Splits text into word and punctuation tokens using a regex.

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

    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into a list of non-whitespace tokens.

        Whitespace-only fragments produced by the split are dropped, and
        every remaining token is stripped of leading/trailing whitespace.
        """
        pieces = self._pattern.split(text)
        return [piece.strip() for piece in pieces if piece.strip()]


def iter_tokens(
    texts: Iterable[str], tokenizer: SimpleTokenizer | None = None
) -> Iterator[str]:
    """Tokenize each chunk of text in ``texts``, yielding tokens one at a time.

    ``texts`` can be any iterable of text chunks -- a list held in memory, a
    generator reading files lazily, or anything else. Chunks are consumed
    one at a time and each one's tokens are yielded before moving on to the
    next, so this never needs more than one chunk of text in memory at once.
    """
    tokenizer = tokenizer or SimpleTokenizer()
    for text in texts:
        yield from tokenizer.tokenize(text)


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
