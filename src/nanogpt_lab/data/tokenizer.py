"""A minimal character-level tokenizer, built from scratch.

Char-level keeps the vocabulary tiny (65 symbols for Tiny Shakespeare) and needs
no external tokenizer library — the whole thing is a pair of dicts. Unknown
characters at inference are skipped so the generation path never throws.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.itos: dict[int, str] = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        return [stoi[c] for c in text if c in stoi]

    def decode(self, ids: list[int]) -> str:
        itos = self.itos
        return "".join(itos[i] for i in ids if i in itos)

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> CharTokenizer:
        return cls(data["chars"])
