"""Raw text -> Tokenizer -> token IDs.

Each character is one token. The vocabulary is every unique character
in the training text, sorted so the same text always gets the same ids.
"""

import json
from pathlib import Path


def read_text(path: str) -> str:
    """Load the raw training text."""
    text = Path(path).read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Training text is empty: {path}")
    return text


class Tokenizer:
    """Map characters to integer ids, and ids back to characters."""

    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}

    def fit(self, text: str) -> None:
        """Build the vocabulary from raw text."""
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Turn raw text into token ids."""
        if not self.stoi:
            raise RuntimeError("Call fit() before encode().")
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            shown = ", ".join(repr(ch) for ch in unknown)
            raise ValueError(f"Characters not in vocabulary: {shown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Turn token ids back into text."""
        if not self.itos:
            raise RuntimeError("Call fit() or load() before decode().")
        return "".join(self.itos[i] for i in token_ids)

    def save(self, path: str) -> None:
        """Write the character list so later steps can reload the same ids."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chars": self.chars}
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.chars = list(data["chars"])
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        return tokenizer
