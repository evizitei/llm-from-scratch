from pathlib import Path

from llm_from_scratch.data.tokenizer import (
    SimpleTokenizer,
    iter_directory_texts,
    iter_tokens,
)
from llm_from_scratch.data.vocabulary import build_vocabulary


def test_splits_words_and_commas_and_periods():
    tokenizer = SimpleTokenizer()
    text = "Hello, world. This, is a test."

    result = tokenizer.tokenize(text)

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

    result = tokenizer.tokenize(text)

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


def test_tokenizes_the_verdict_into_4690_tokens(the_verdict_text):
    tokenizer = SimpleTokenizer()

    result = tokenizer.tokenize(the_verdict_text)

    assert len(result) == 4690


def test_first_thirty_tokens_of_the_verdict(the_verdict_text):
    tokenizer = SimpleTokenizer()

    result = tokenizer.tokenize(the_verdict_text)

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


def test_iter_tokens_flattens_tokenized_chunks_of_in_memory_text():
    texts = ["Hello, world.", "Another chunk!"]

    tokens = list(iter_tokens(texts))

    assert tokens == ["Hello", ",", "world", ".", "Another", "chunk", "!"]


def test_iter_tokens_uses_the_given_tokenizer():
    class UpperTokenizer(SimpleTokenizer):
        def tokenize(self, text: str) -> list[str]:
            return [token.upper() for token in super().tokenize(text)]

    tokens = list(iter_tokens(["hello world"], tokenizer=UpperTokenizer()))

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


def test_iter_tokens_output_can_feed_build_vocabulary():
    # iter_tokens produces a plain iterator over strings, which is all
    # build_vocabulary (from the independent vocabulary module) needs.
    texts = ["the cat sat.", "the dog sat."]

    vocabulary = build_vocabulary(iter_tokens(texts))

    assert len(vocabulary) == len({"the", "cat", "sat", ".", "dog"})
    assert "cat" in vocabulary
    assert "bird" not in vocabulary


def test_iter_directory_texts_output_can_feed_iter_tokens_and_build_vocabulary(
    tmp_path: Path,
):
    (tmp_path / "doc1.txt").write_text("Hello, world.")
    (tmp_path / "doc2.txt").write_text("Hello again, world.")

    vocabulary = build_vocabulary(iter_tokens(iter_directory_texts(tmp_path)))

    for token in ["Hello", "world", "again", ",", "."]:
        assert token in vocabulary
    assert vocabulary.token_to_id("again") < vocabulary.token_to_id("world")
