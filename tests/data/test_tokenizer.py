from llm_from_scratch.data.tokenizer import SimpleTokenizer


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
