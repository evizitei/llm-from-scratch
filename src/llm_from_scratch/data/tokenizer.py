"""A minimal, regex-based tokenizer.

This mirrors the "toy" tokenizer built up in chapter 2 of *Build a Large
Language Model (From Scratch)*: text is split on whitespace and a fixed set
of punctuation characters, and the resulting whitespace-only fragments are
dropped. It has no notion of a vocabulary yet -- it just turns raw text into
a list of string tokens.

This module intentionally contains only the tokenization logic itself. Any
code that downloads sample text, prints results, or otherwise deals with the
outside world belongs elsewhere (see ``tests/data/test_tokenizer.py`` and
``viz/components/tokenizer/`` for that).
"""

from __future__ import annotations

import re

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
