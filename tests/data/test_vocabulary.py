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


# --- encoding -------------------------------------------------------------


def test_encode_maps_known_tokens_to_their_ids():
    vocabulary = build_vocabulary(iter(["the", "cat"]))

    ids = list(vocabulary.encode(iter(["the", "cat", "the"])))

    assert ids == [
        vocabulary.token_to_id("the"),
        vocabulary.token_to_id("cat"),
        vocabulary.token_to_id("the"),
    ]


def test_encode_substitutes_the_unknown_token_for_unseen_tokens():
    vocabulary = build_vocabulary(iter(["the", "cat"]))

    ids = list(vocabulary.encode(iter(["the", "hippopotamus"])))

    assert ids == [vocabulary.token_to_id("the"), vocabulary.unknown_id]


def test_encode_raises_for_unseen_tokens_without_an_unknown_token():
    vocabulary = build_vocabulary(iter(["the"]), special_tokens=())

    with pytest.raises(KeyError):
        list(vocabulary.encode(iter(["the", "hippopotamus"])))


def test_encode_is_lazy():
    # Nothing is pulled from the input until the output is pulled from,
    # which is what lets a corpus larger than memory stream through.
    pulled = []

    def tokens():
        for token in ["a", "b", "c"]:
            pulled.append(token)
            yield token

    vocabulary = build_vocabulary(iter(["a", "b", "c"]))
    ids = vocabulary.encode(tokens())

    assert pulled == []
    next(ids)
    assert pulled == ["a"]


# --- decoding -------------------------------------------------------------


def test_decode_maps_ids_back_to_tokens():
    vocabulary = build_vocabulary(iter(["the", "cat"]))
    ids = vocabulary.encode(iter(["the", "cat"]))

    assert list(vocabulary.decode(ids)) == ["the", "cat"]


def test_decode_round_trips_encode_with_unknowns_collapsed():
    vocabulary = build_vocabulary(iter(["the", "cat", "sat"]))

    tokens = ["the", "aardvark", "sat"]
    decoded = list(vocabulary.decode(vocabulary.encode(iter(tokens))))

    assert decoded == ["the", UNKNOWN_TOKEN, "sat"]


def test_decode_can_skip_special_tokens():
    vocabulary = build_vocabulary(iter(["the", "cat"]))
    tokens = ["the", END_OF_TEXT_TOKEN, "aardvark", "cat"]

    ids = list(vocabulary.encode(iter(tokens)))

    assert list(vocabulary.decode(ids, skip_special_tokens=True)) == ["the", "cat"]


def test_decode_raises_for_an_id_outside_the_vocabulary():
    vocabulary = build_vocabulary(iter(["the"]))

    with pytest.raises(KeyError):
        list(vocabulary.decode(iter([len(vocabulary)])))


def test_decode_is_lazy():
    pulled = []

    def ids():
        for id_ in [0, 1]:
            pulled.append(id_)
            yield id_

    vocabulary = build_vocabulary(iter(["a", "b"]))
    tokens = vocabulary.decode(ids())

    assert pulled == []
    next(tokens)
    assert pulled == [0]
