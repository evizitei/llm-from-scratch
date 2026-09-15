"""Tokenizers: text <-> tokens, and (given a vocabulary) tokens <-> ids.

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

``VocabularyTokenizer`` is the production-facing pairing of the two: hand
it a ``Vocabulary`` and it encodes text straight through to ids and decodes
ids back to text, streaming in both directions. ``SimpleTokenizer`` is what
you use to *build* a vocabulary in the first place; ``VocabularyTokenizer``
is what you use once you have one.

The dependency between the two modules points one way only -- a tokenizer
knows about vocabularies, a vocabulary knows nothing about tokenizers.
``Vocabulary`` is a lookup table; deciding to substitute ``<|unk|>`` for an
unseen word, and knowing how to glue tokens back into readable text, are
both facts about how text was split, so they live here.

This module also contains only tokenization logic itself. Any code that
downloads sample text, prints results, or otherwise deals with the outside
world belongs elsewhere (see ``tests/data/test_tokenizer.py`` and
``viz/components/tokenizer/`` for that).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from llm_from_scratch.data.vocabulary import Vocabulary

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


# Punctuation that hugs the word before it (no space in front) and
# punctuation that hugs the word after it (no space behind), used to glue
# tokens back into readable text. Splitting threw the original whitespace
# away, so this is a reconstruction, not a recovery -- see
# VocabularyTokenizer.decode.
#
# Curly quotes are directional, so they land in whichever set they belong
# to. The straight apostrophe is in both on purpose: it far more often
# joins a word ("don't", "Bennet's") than opens a quotation, and being in
# both sets is what keeps those contractions intact. The straight double
# quote is genuinely ambiguous and gets tracked separately below.
_NO_SPACE_BEFORE = frozenset(",.:;?_!)'”’")
_NO_SPACE_AFTER = frozenset("('“‘")
_DOUBLE_QUOTE = '"'


class VocabularyTokenizer:
    """Encodes text to token ids and back, against a fixed ``Vocabulary``.

    This is the tokenizer you use once a vocabulary exists: the pairing of
    a splitting rule (how text becomes tokens) with a lookup table (what id
    each token has). Both directions stream, one item at a time, so a
    corpus far larger than memory can pass through.

    Parameters
    ----------
    vocabulary:
        The token <-> id table to encode against. Its ``<|unk|>`` entry, if
        it has one, is what unseen tokens become.
    tokenizer:
        How text is split into tokens. Defaults to a plain
        ``SimpleTokenizer``; pass the *same* splitting rule that built
        ``vocabulary``, or tokens will come out that the table has never
        seen and collapse into ``<|unk|>``.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()

    @property
    def vocabulary(self) -> Vocabulary:
        """The vocabulary this tokenizer encodes against."""
        return self._vocabulary

    def encode(self, texts: Iterable[str]) -> Iterator[int]:
        """Stream chunks of text into token ids, one id at a time.

        Splits each chunk with this tokenizer's splitting rule and looks
        every token up. Unseen tokens become ``<|unk|>``; see
        ``encode_tokens`` for what happens when the vocabulary has no such
        entry.
        """
        return self.encode_tokens(self._tokenizer.tokenize(texts))

    def encode_tokens(self, tokens: Iterable[str]) -> Iterator[int]:
        """Stream already-split tokens into their ids, one at a time.

        Tokens that aren't in the vocabulary become ``<|unk|>``'s id. If
        the vocabulary has no ``<|unk|>`` entry there is nothing sensible
        to substitute, so an unknown token raises ``KeyError`` instead --
        silently dropping it would quietly corrupt the stream.
        """
        vocabulary = self._vocabulary
        unknown_id = vocabulary.unknown_id
        for token in tokens:
            if unknown_id is None:
                yield vocabulary.token_to_id(token)
            elif token in vocabulary:
                yield vocabulary[token]
            else:
                yield unknown_id

    def decode_to_tokens(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> Iterator[str]:
        """Stream ids back into their tokens, one token at a time.

        An id that isn't in the vocabulary raises ``KeyError``: unlike an
        unseen word, an out-of-range id is a bug (a mismatched vocabulary,
        an off-by-one in a model's output layer) rather than data, and is
        worth failing loudly on.

        Pass ``skip_special_tokens=True`` to drop reserved tokens from the
        output -- the usual choice when showing generated text to a human,
        who has no use for seeing ``<|endoftext|>``.
        """
        vocabulary = self._vocabulary
        for id_ in ids:
            if skip_special_tokens and vocabulary.is_special_id(id_):
                continue
            yield vocabulary.id_to_token(id_)

    def decode(
        self,
        ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> Iterator[str]:
        """Stream ids back into text fragments, ready to ``"".join``.

        Yields one fragment per token, each carrying its own leading space
        (or not), so the caller can join them without buffering the whole
        text::

            "".join(tokenizer.decode(ids))

        Note this is a *reconstruction*, not an exact inverse of
        ``encode``. Splitting discarded the original whitespace, so spacing
        is re-derived from a rule of thumb: punctuation like ``,`` and
        ``.`` hugs the word before it, ``(`` and opening quotes hug the
        word after, everything else gets a space in front. Doubled spaces,
        line breaks and unusual punctuation don't survive the round trip --
        and any word that hit ``<|unk|>`` on the way in is gone for good.
        Lossy detokenization like this is inherent to splitting text into
        words, and is one more reason production tokenizers work at the
        byte level instead.
        """
        first = True
        hugs_next = False
        double_quote_open = False
        for token in self.decode_to_tokens(ids, skip_special_tokens):
            if token == _DOUBLE_QUOTE:
                # A straight " is an opening quote if none is outstanding
                # and a closing one otherwise, which is the only way to
                # tell the two apart once the whitespace is gone.
                opening = not double_quote_open
                double_quote_open = opening
                space_before = opening
                hugs_after = opening
            else:
                space_before = token not in _NO_SPACE_BEFORE
                hugs_after = token in _NO_SPACE_AFTER

            yield " " + token if space_before and not first and not hugs_next else token
            first = False
            hugs_next = hugs_after


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
