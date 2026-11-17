import pytest
from minigpt.tokenizer import CharTokenizer


def test_vocab_size_and_determinism():
    t1 = CharTokenizer("cbaabc")
    t2 = CharTokenizer("abc")

    assert t1.vocab_size == 3
    assert t1.stoi == {"a": 0, "b": 1, "c": 2}
    assert t1.itos == {0: "a", 1: "b", 2: "c"}
    assert t1.stoi == t2.stoi


def test_encode_decode():
    corpus = "Привет, мир!\nHello world 123."
    tokenizer = CharTokenizer(corpus)

    text = "Привет, world 123!"
    encoded = tokenizer.encode(text)

    assert isinstance(encoded, list)
    assert all(isinstance(token, int) for token in encoded)
    assert tokenizer.decode(encoded) == text


def test_empty_strings():
    tokenizer = CharTokenizer("abc")
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

    empty_tokenizer = CharTokenizer("")
    assert empty_tokenizer.vocab_size == 0
    assert empty_tokenizer.encode("") == []


def test_unknown_char_raises():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.encode("abcd")


def test_unknown_token_id_raises():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.decode([0, 99])