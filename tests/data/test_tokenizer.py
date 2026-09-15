from pathlib import Path

import pytest

from llm_from_scratch.data.tokenizer import (
    SimpleTokenizer,
    VocabularyTokenizer,
    iter_directory_texts,
)
from llm_from_scratch.data.vocabulary import (
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    build_vocabulary,
)


def _tokenizer_for(*texts: str, special_tokens=None) -> VocabularyTokenizer:
    """A VocabularyTokenizer over a vocabulary built from ``texts``."""
    kwargs = {} if special_tokens is None else {"special_tokens": special_tokens}
    vocabulary = build_vocabulary(SimpleTokenizer().tokenize(texts), **kwargs)
    return VocabularyTokenizer(vocabulary)


def test_splits_words_and_commas_and_periods():
    tokenizer = SimpleTokenizer()
    text = "Hello, world. This, is a test."

    result = list(tokenizer.tokenize([text]))

    assert result == [
        "Hello",
        ",",
        "world",
        ".",
        "This",
        ",",
        "is",
        "a",
        "test",
        ".",
    ]


def test_handles_question_marks_and_double_dashes():
    tokenizer = SimpleTokenizer()
    text = "Hello, world. Is this-- a test?"

    result = list(tokenizer.tokenize([text]))

    assert result == [
        "Hello",
        ",",
        "world",
        ".",
        "Is",
        "this",
        "--",
        "a",
        "test",
        "?",
    ]


def test_handles_curly_quotation_marks():
    # Pride and Prejudice (and plenty of other real-world text) uses curly
    # "smart" quotes rather than straight ones -- the tokenizer needs to
    # split them off as their own tokens too, not just the straight ' and "
    # that the default pattern already knows about.
    tokenizer = SimpleTokenizer()
    text = "“Impossible, Mr. Bennet, impossible!”"

    result = list(tokenizer.tokenize([text]))

    assert result == [
        "“",
        "Impossible",
        ",",
        "Mr",
        ".",
        "Bennet",
        ",",
        "impossible",
        "!",
        "”",
    ]


def test_tokenizes_the_verdict_into_4690_tokens(the_verdict_text):
    tokenizer = SimpleTokenizer()

    result = list(tokenizer.tokenize([the_verdict_text]))

    assert len(result) == 4690


def test_first_thirty_tokens_of_the_verdict(the_verdict_text):
    tokenizer = SimpleTokenizer()

    result = list(tokenizer.tokenize([the_verdict_text]))

    assert result[:30] == [
        "I",
        "HAD",
        "always",
        "thought",
        "Jack",
        "Gisburn",
        "rather",
        "a",
        "cheap",
        "genius",
        "--",
        "though",
        "a",
        "good",
        "fellow",
        "enough",
        "--",
        "so",
        "it",
        "was",
        "no",
        "great",
        "surprise",
        "to",
        "me",
        "to",
        "hear",
        "that",
        ",",
        "in",
    ]


def test_tokenize_flattens_tokens_across_multiple_chunks_of_text():
    texts = ["Hello, world.", "Another chunk!"]

    tokens = list(SimpleTokenizer().tokenize(texts))

    assert tokens == ["Hello", ",", "world", ".", "Another", "chunk", "!"]


def test_tokenize_pulls_one_chunk_at_a_time():
    # A generator that records the order chunks are pulled in demonstrates
    # that tokenize() streams -- it never asks for chunk N+1 before it has
    # finished yielding every token from chunk N.
    pulled = []

    def texts():
        for text in ["one two", "three four"]:
            pulled.append(text)
            yield text

    tokens = SimpleTokenizer().tokenize(texts())

    assert next(tokens) == "one"
    assert pulled == ["one two"]
    assert next(tokens) == "two"
    assert pulled == ["one two"]
    assert next(tokens) == "three"
    assert pulled == ["one two", "three four"]


def test_tokenize_can_be_subclassed_to_transform_tokens():
    class UpperTokenizer(SimpleTokenizer):
        def tokenize(self, texts):
            for token in super().tokenize(texts):
                yield token.upper()

    tokens = list(UpperTokenizer().tokenize(["hello world"]))

    assert tokens == ["HELLO", "WORLD"]


def test_iter_directory_texts_yields_one_chunk_per_file(tmp_path: Path):
    (tmp_path / "b.txt").write_text("second file")
    (tmp_path / "a.txt").write_text("first file")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "ignored.txt").write_text("should not be read")

    texts = list(iter_directory_texts(tmp_path))

    # Visited in name order, and the subdirectory is skipped entirely.
    assert texts == ["first file", "second file"]


def test_tokenize_output_can_feed_build_vocabulary():
    # tokenize() produces a plain iterator over strings, which is all
    # build_vocabulary (from the independent vocabulary module) needs.
    texts = ["the cat sat.", "the dog sat."]

    vocabulary = build_vocabulary(SimpleTokenizer().tokenize(texts), special_tokens=())

    assert len(vocabulary) == len({"the", "cat", "sat", ".", "dog"})
    assert "cat" in vocabulary
    assert "bird" not in vocabulary


def test_iter_directory_texts_output_can_feed_tokenize_and_build_vocabulary(
    tmp_path: Path,
):
    (tmp_path / "doc1.txt").write_text("Hello, world.")
    (tmp_path / "doc2.txt").write_text("Hello again, world.")

    vocabulary = build_vocabulary(
        SimpleTokenizer().tokenize(iter_directory_texts(tmp_path))
    )

    for token in ["Hello", "world", "again", ",", "."]:
        assert token in vocabulary
    assert vocabulary.token_to_id("again") < vocabulary.token_to_id("world")


# --- VocabularyTokenizer: encoding ----------------------------------------


def test_encode_maps_text_to_the_vocabulary_ids():
    tokenizer = _tokenizer_for("the cat sat")
    vocabulary = tokenizer.vocabulary

    ids = list(tokenizer.encode(["the cat"]))

    assert ids == [vocabulary.token_to_id("the"), vocabulary.token_to_id("cat")]


def test_encode_splits_punctuation_the_same_way_the_vocabulary_was_built():
    tokenizer = _tokenizer_for("the cat sat.")

    tokens = list(tokenizer.decode_to_tokens(tokenizer.encode(["the cat sat."])))

    assert tokens == ["the", "cat", "sat", "."]


def test_encode_substitutes_the_unknown_token_for_unseen_words():
    tokenizer = _tokenizer_for("the cat sat")

    ids = list(tokenizer.encode(["the hippopotamus"]))

    assert ids == [
        tokenizer.vocabulary.token_to_id("the"),
        tokenizer.vocabulary.unknown_id,
    ]


def test_encode_raises_for_unseen_words_without_an_unknown_token():
    tokenizer = _tokenizer_for("the cat", special_tokens=())

    with pytest.raises(KeyError):
        list(tokenizer.encode(["the hippopotamus"]))


def test_encode_tokens_skips_splitting_for_an_already_tokenized_stream():
    tokenizer = _tokenizer_for("the cat sat")

    # "the cat" as one string would be split into two tokens; as a single
    # pre-split token it is simply unknown.
    ids = list(tokenizer.encode_tokens(iter(["the cat"])))

    assert ids == [tokenizer.vocabulary.unknown_id]


def test_encode_is_lazy():
    # Nothing is pulled from the input until the output is pulled from,
    # which is what lets a corpus larger than memory stream through.
    pulled = []

    def texts():
        for text in ["a", "b", "c"]:
            pulled.append(text)
            yield text

    tokenizer = _tokenizer_for("a b c")
    ids = tokenizer.encode(texts())

    assert pulled == []
    next(ids)
    assert pulled == ["a"]


# --- VocabularyTokenizer: decoding ----------------------------------------


def test_decode_reconstructs_the_original_text():
    text = "Hello, world. This is a test."
    tokenizer = _tokenizer_for(text)

    assert "".join(tokenizer.decode(tokenizer.encode([text]))) == text


def test_decode_reconstructs_spacing_around_quotes_and_brackets():
    text = 'She said "yes" and (then) left; he didn\'t.'
    tokenizer = _tokenizer_for(text)

    assert "".join(tokenizer.decode(tokenizer.encode([text]))) == text


def test_decode_reconstructs_spacing_around_curly_quotes():
    text = "“Impossible, Mr. Bennet, impossible!”"
    tokenizer = _tokenizer_for(text)

    assert "".join(tokenizer.decode(tokenizer.encode([text]))) == text


def test_decode_cannot_recover_words_that_hit_the_unknown_token():
    # The round trip is lossy by construction: once a word has collapsed
    # into <|unk|>, nothing downstream can tell what it used to be.
    tokenizer = _tokenizer_for("the cat sat")

    text = "".join(tokenizer.decode(tokenizer.encode(["the hippopotamus"])))

    assert text == f"the {UNKNOWN_TOKEN}"


def test_decode_to_tokens_yields_tokens_rather_than_text():
    tokenizer = _tokenizer_for("the cat sat.")

    tokens = list(tokenizer.decode_to_tokens(tokenizer.encode(["the cat."])))

    assert tokens == ["the", "cat", "."]


def test_decode_can_skip_special_tokens():
    tokenizer = _tokenizer_for("the cat")
    ids = tokenizer.encode_tokens(iter(["the", END_OF_TEXT_TOKEN, "aardvark", "cat"]))

    assert "".join(tokenizer.decode(ids, skip_special_tokens=True)) == "the cat"


def test_decode_raises_for_an_id_outside_the_vocabulary():
    # Unlike an unseen word, an out-of-range id is a bug rather than data.
    tokenizer = _tokenizer_for("the cat")

    with pytest.raises(KeyError):
        list(tokenizer.decode(iter([len(tokenizer.vocabulary)])))


def test_decode_is_lazy():
    pulled = []

    def ids():
        for id_ in [0, 1]:
            pulled.append(id_)
            yield id_

    tokenizer = _tokenizer_for("a b")
    tokens = tokenizer.decode(ids())

    assert pulled == []
    next(tokens)
    assert pulled == [0]


def test_vocabulary_tokenizer_accepts_a_custom_splitting_rule():
    # The splitting rule is injected, so a vocabulary built with one
    # tokenizer can be encoded against with the very same one.
    splitter = SimpleTokenizer()
    vocabulary = build_vocabulary(splitter.tokenize(["the cat sat"]))

    tokenizer = VocabularyTokenizer(vocabulary, tokenizer=splitter)

    assert list(tokenizer.encode(["the cat"])) == [
        vocabulary.token_to_id("the"),
        vocabulary.token_to_id("cat"),
    ]
