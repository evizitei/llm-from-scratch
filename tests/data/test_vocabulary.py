import pytest

from llm_from_scratch.data.vocabulary import (
    DEFAULT_SPECIAL_TOKENS,
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    Vocabulary,
    build_vocabulary,
)


def test_build_vocabulary_returns_a_vocabulary():
    assert isinstance(build_vocabulary(iter(["a"])), Vocabulary)


def test_build_vocabulary_assigns_ids_in_alphabetical_order():
    tokens = iter(["banana", "apple", "cherry", "apple"])

    vocabulary = build_vocabulary(tokens, special_tokens=())

    assert vocabulary.token_to_id("apple") == 0
    assert vocabulary.token_to_id("banana") == 1
    assert vocabulary.token_to_id("cherry") == 2


def test_build_vocabulary_deduplicates_tokens():
    tokens = iter(["a", "b", "a", "b", "a"])

    vocabulary = build_vocabulary(tokens, special_tokens=())

    assert len(vocabulary) == 2


def test_build_vocabulary_only_needs_an_iterator_over_strings():
    # A generator is a plain iterator over strings, nothing more -- this is
    # the entire contract build_vocabulary relies on. No tokenizer, no
    # files, no knowledge of where these strings came from.
    def tokens():
        yield "z"
        yield "a"

    vocabulary = build_vocabulary(tokens(), special_tokens=())

    assert [vocabulary.id_to_token(i) for i in range(len(vocabulary))] == ["a", "z"]


def test_vocabulary_len_and_contains():
    vocabulary = build_vocabulary(iter(["one", "two", "three"]), special_tokens=())

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


# --- special tokens -------------------------------------------------------


def test_build_vocabulary_reserves_special_tokens_by_default():
    vocabulary = build_vocabulary(iter(["word"]))

    assert vocabulary.special_tokens == DEFAULT_SPECIAL_TOKENS
    assert UNKNOWN_TOKEN in vocabulary
    assert END_OF_TEXT_TOKEN in vocabulary
    assert len(vocabulary) == 1 + len(DEFAULT_SPECIAL_TOKENS)


def test_special_tokens_take_the_lowest_ids():
    # Low ids keep specials stable as the corpus grows, which matters
    # because an id is a row index into a model's embedding matrix.
    vocabulary = build_vocabulary(iter(["aardvark", "zebra"]))

    assert vocabulary.token_to_id(UNKNOWN_TOKEN) == 0
    assert vocabulary.token_to_id(END_OF_TEXT_TOKEN) == 1
    assert vocabulary.token_to_id("aardvark") == 2
    assert vocabulary.token_to_id("zebra") == 3


def test_special_token_ids_do_not_shift_when_the_corpus_grows():
    small = build_vocabulary(iter(["b"]))
    large = build_vocabulary(iter(["a", "b", "c"]))

    assert small.unknown_id == large.unknown_id


def test_build_vocabulary_accepts_custom_special_tokens():
    vocabulary = build_vocabulary(iter(["word"]), special_tokens=["<|pad|>"])

    assert vocabulary.special_tokens == ("<|pad|>",)
    assert vocabulary.token_to_id("<|pad|>") == 0
    assert UNKNOWN_TOKEN not in vocabulary


def test_build_vocabulary_does_not_duplicate_a_special_token_seen_in_the_corpus():
    vocabulary = build_vocabulary(iter(["word", END_OF_TEXT_TOKEN]))

    assert len(vocabulary) == 1 + len(DEFAULT_SPECIAL_TOKENS)
    assert vocabulary.token_to_id(END_OF_TEXT_TOKEN) == 1


def test_unknown_id_is_none_without_an_unknown_token():
    assert build_vocabulary(iter(["word"]), special_tokens=()).unknown_id is None


def test_is_special_id_distinguishes_reserved_tokens_from_words():
    vocabulary = build_vocabulary(iter(["word"]))

    assert vocabulary.is_special_id(vocabulary.token_to_id(UNKNOWN_TOKEN))
    assert vocabulary.is_special_id(vocabulary.token_to_id(END_OF_TEXT_TOKEN))
    assert not vocabulary.is_special_id(vocabulary.token_to_id("word"))
