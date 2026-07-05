"""Character tokenizer: roundtrip, unknown handling, vocab, (de)serialization."""

from __future__ import annotations

from nanogpt_lab.data.tokenizer import CharTokenizer


def test_roundtrip() -> None:
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_unknown_characters_are_skipped() -> None:
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("aXbY") == [tok.stoi["a"], tok.stoi["b"]]


def test_vocab_is_sorted_unique() -> None:
    tok = CharTokenizer.from_text("cba cba")
    assert tok.vocab_size == 4  # {' ', 'a', 'b', 'c'}
    assert tok.chars == sorted(tok.chars)


def test_dict_roundtrip() -> None:
    tok = CharTokenizer.from_text("the quick brown fox")
    restored = CharTokenizer.from_dict(tok.to_dict())
    assert restored.chars == tok.chars
    assert restored.stoi == tok.stoi
