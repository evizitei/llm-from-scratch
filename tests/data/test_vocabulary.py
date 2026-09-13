import pytest

from llm_from_scratch.data.vocabulary import Vocabulary, build_vocabulary


def test_build_vocabulary_returns_a_vocabulary():
    assert isinstance(build_vocabulary(iter(["a"])), Vocabulary)


def test_build_vocabulary_assigns_ids_in_alphabetical_order():
    tokens = iter(["banana", "apple", "cherry", "apple"])

    vocabulary = build_vocabulary(tokens)

    assert vocabulary.token_to_id("apple") == 0
    assert vocabulary.token_to_id("banana") == 1
    assert vocabulary.token_to_id("cherry") == 2


def test_build_vocabulary_deduplicates_tokens():
    tokens = iter(["a", "b", "a", "b", "a"])

    vocabulary = build_vocabulary(tokens)

    assert len(vocabulary) == 2


def test_build_vocabulary_only_needs_an_iterator_over_strings():
    # A generator is a plain iterator over strings, nothing more -- this is
    # the entire contract build_vocabulary relies on. No tokenizer, no
    # files, no knowledge of where these strings came from.
    def tokens():
        yield "z"
        yield "a"

    vocabulary = build_vocabulary(tokens())

    assert [vocabulary.id_to_token(i) for i in range(len(vocabulary))] == ["a", "z"]


def test_vocabulary_len_and_contains():
    vocabulary = build_vocabulary(iter(["one", "two", "three"]))

    assert len(vocabulary) == 3
    assert "two" in vocabulary
    assert "four" not in vocabulary


def test_vocabulary_getitem_matches_token_to_id():
    vocabulary = build_vocabulary(iter(["one", "two"]))

    assert vocabulary["one"] == vocabulary.token_to_id("one")


def test_vocabulary_id_to_token_round_trips():
    vocabulary = build_vocabulary(iter(["one", "two", "three"]))

    for token in ["one", "two", "three"]:
        token_id = vocabulary.token_to_id(token)
        assert vocabulary.id_to_token(token_id) == token


def test_vocabulary_raises_key_error_for_unknown_token():
    vocabulary = build_vocabulary(iter(["one"]))

    with pytest.raises(KeyError):
        vocabulary.token_to_id("missing")

    with pytest.raises(KeyError):
        vocabulary["missing"]
