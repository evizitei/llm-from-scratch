from pathlib import Path

from llm_from_scratch.data.tokenizer import SimpleTokenizer, iter_directory_texts
from llm_from_scratch.data.vocabulary import build_vocabulary


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

    vocabulary = build_vocabulary(SimpleTokenizer().tokenize(iter_directory_texts(tmp_path)))

    for token in ["Hello", "world", "again", ",", "."]:
        assert token in vocabulary
    assert vocabulary.token_to_id("again") < vocabulary.token_to_id("world")
