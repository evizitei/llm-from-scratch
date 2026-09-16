import pytest

from llm_from_scratch.data.sampler import (
    ContextPair,
    DataSampler,
    TokenWindow,
    VocabularyEncoder,
)
from llm_from_scratch.data.tokenizer import SimpleTokenizer, VocabularyTokenizer
from llm_from_scratch.data.vocabulary import build_vocabulary


class StubEncoder:
    """A TextEncoder that maps each character to its position in a word.

    Deliberately trivial: these tests are about how pairs are built, not
    about any particular tokenization, so the "tokens" here are single
    characters and the ids are their code points.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids) -> str:
        return "".join(chr(id_) for id_ in ids)


def _encoder_for(*texts: str) -> VocabularyEncoder:
    """A VocabularyEncoder over a vocabulary built from ``texts``."""
    vocabulary = build_vocabulary(SimpleTokenizer().tokenize(texts))
    return VocabularyEncoder(VocabularyTokenizer(vocabulary))


def test_yields_growing_contexts_with_the_next_token_as_target():
    sampler = DataSampler(StubEncoder())

    pairs = list(sampler.sample("abcd"))

    assert pairs == [
        ((97,), 98),
        ((97, 98), 99),
        ((97, 98, 99), 100),
    ]


def test_pairs_unpack_as_context_and_target():
    sampler = DataSampler(StubEncoder())

    context, target = next(iter(sampler.sample("ab")))

    assert context == (97,)
    assert target == 98


def test_a_single_string_is_treated_as_one_chunk_not_a_stream():
    sampler = DataSampler(StubEncoder())

    # "ab" iterated as a sequence would be two one-character chunks, and
    # neither of those is long enough to make a pair.
    assert list(sampler.sample("ab")) == [ContextPair((97,), 98)]


def test_context_does_not_run_across_chunk_boundaries():
    sampler = DataSampler(StubEncoder())

    pairs = list(sampler.sample(["ab", "cd"]))

    assert pairs == [
        ((97,), 98),
        ((99,), 100),
    ]


def test_chunks_too_short_to_pair_yield_nothing():
    sampler = DataSampler(StubEncoder())

    assert list(sampler.sample(["", "a", "ab"])) == [ContextPair((97,), 98)]


def test_consumes_the_text_stream_lazily():
    sampler = DataSampler(StubEncoder())
    pulled = []

    def texts():
        for text in ["ab", "cd"]:
            pulled.append(text)
            yield text

    pairs = sampler.sample(texts())
    next(pairs)

    assert pulled == ["ab"]


def test_encodes_with_a_vocabulary_tokenizer():
    encoder = _encoder_for("the cat sat")
    sampler = DataSampler(encoder)

    pairs = list(sampler.sample("the cat sat"))

    ids = encoder.encode("the cat sat")
    assert len(ids) == 3
    assert pairs == [
        ((ids[0],), ids[1]),
        ((ids[0], ids[1]), ids[2]),
    ]


def test_decode_pair_renders_context_and_target_as_text():
    sampler = DataSampler(_encoder_for("the cat sat on the mat."))

    pairs = list(sampler.sample("the cat sat"))

    # Each side is decoded on its own, so the target has no leading space
    # here -- a word-level decode never puts one in front of the first
    # token it is handed.
    assert sampler.decode_pair(pairs[-1]) == ("the cat", "sat")


def test_works_with_a_tiktoken_bpe_encoder():
    tiktoken = pytest.importorskip("tiktoken")
    sampler = DataSampler(tiktoken.get_encoding("gpt2"))

    pairs = list(sampler.sample("Hello, world!"))

    assert len(pairs) == len(tiktoken.get_encoding("gpt2").encode("Hello, world!")) - 1
    assert all(
        pair.context[: len(pairs[0].context)] == pairs[0].context for pair in pairs
    )
    context_text, target_text = sampler.decode_pair(pairs[0])
    assert context_text + target_text == "Hello,"


def test_max_length_caps_and_then_slides_the_context():
    sampler = DataSampler(StubEncoder(), max_length=2)

    pairs = list(sampler.sample("abcd"))

    assert pairs == [
        ((97,), 98),
        ((97, 98), 99),
        ((98, 99), 100),
    ]


def test_max_length_must_be_positive():
    with pytest.raises(ValueError, match="max_length"):
        DataSampler(StubEncoder(), max_length=0)


def test_stride_must_be_positive():
    with pytest.raises(ValueError, match="stride"):
        DataSampler(StubEncoder(), max_length=2, stride=0)


def test_window_needs_a_max_length():
    sampler = DataSampler(StubEncoder())

    with pytest.raises(ValueError, match="max_length"):
        next(sampler.window("abcd"))


def test_stride_defaults_to_max_length_and_tiles_without_overlap():
    sampler = DataSampler(StubEncoder(), max_length=2)

    windows = list(sampler.window("abcdef"))

    assert sampler.stride == 2
    assert windows == [
        ((97, 98), (98, 99)),
        ((99, 100), (100, 101)),
    ]


def test_a_smaller_stride_overlaps_windows():
    sampler = DataSampler(StubEncoder(), max_length=2, stride=1)

    windows = list(sampler.window("abcd"))

    assert windows == [
        ((97, 98), (98, 99)),
        ((98, 99), (99, 100)),
    ]


def test_a_larger_stride_skips_text_between_windows():
    sampler = DataSampler(StubEncoder(), max_length=2, stride=4)

    windows = list(sampler.window("abcdefg"))

    assert windows == [
        ((97, 98), (98, 99)),
        ((101, 102), (102, 103)),
    ]


def test_windows_unpack_as_inputs_and_targets():
    sampler = DataSampler(StubEncoder(), max_length=2)

    inputs, targets = next(sampler.window("abc"))

    assert inputs == (97, 98)
    assert targets == (98, 99)


def test_a_window_holds_every_prefix_pair_inside_it():
    # The point of a window: one of them is the same information as
    # max_length growing-context pairs, which is why overlapping windows
    # duplicate training signal.
    text = "abcd"
    windowed = DataSampler(StubEncoder(), max_length=4)
    unrolled = DataSampler(StubEncoder())

    inputs, targets = next(windowed.window(text + "e"))
    pairs = list(unrolled.sample(text))

    assert [(inputs[: i + 1], targets[i]) for i in range(len(pairs))] == pairs


def test_partial_trailing_windows_are_dropped():
    sampler = DataSampler(StubEncoder(), max_length=3)

    # Seven ids tile into one full window plus a remainder too short to
    # make a second; nothing is padded.
    assert list(sampler.window("abcdefg")) == [
        TokenWindow((97, 98, 99), (98, 99, 100)),
        TokenWindow((100, 101, 102), (101, 102, 103)),
    ]


def test_windows_do_not_straddle_chunk_boundaries():
    sampler = DataSampler(StubEncoder(), max_length=2)

    windows = list(sampler.window(["abc", "def"]))

    assert windows == [
        ((97, 98), (98, 99)),
        ((100, 101), (101, 102)),
    ]


def test_chunks_shorter_than_a_window_yield_nothing():
    sampler = DataSampler(StubEncoder(), max_length=3)

    assert list(sampler.window(["", "ab", "abc"])) == []


def test_window_consumes_the_text_stream_lazily():
    sampler = DataSampler(StubEncoder(), max_length=2)
    pulled = []

    def texts():
        for text in ["abc", "def"]:
            pulled.append(text)
            yield text

    next(sampler.window(texts()))

    assert pulled == ["abc"]


def test_decode_window_renders_inputs_and_targets_as_text():
    sampler = DataSampler(_encoder_for("the cat sat on the mat."), max_length=3)

    window = next(sampler.window("the cat sat on"))

    assert sampler.decode_window(window) == ("the cat sat", "cat sat on")
