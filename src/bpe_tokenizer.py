"""BPE / subword tokenizer for MiniGPT v2.

Wraps the Hugging Face `tokenizers` library. Trained only on TinyStories
training stories. Validation text is never used to fit merges.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class BPETokenizer:
    """Project-facing BPE wrapper with a char-tokenizer-compatible API."""

    def __init__(self, hf_tokenizer: HFTokenizer | None = None, vocab_size: int | None = None):
        self._tokenizer = hf_tokenizer
        self._configured_vocab_size = vocab_size
        self.tokenizer_type = "bpe"

    @property
    def pad_token_id(self) -> int:
        return self._id("<pad>")

    @property
    def unk_token_id(self) -> int:
        return self._id("<unk>")

    @property
    def bos_token_id(self) -> int:
        return self._id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self._id("<eos>")

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")
        return self._tokenizer.get_vocab_size()

    def _id(self, token: str) -> int:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"Special token {token!r} is missing from the vocabulary.")
        return token_id

    def train(self, stories: list[str], vocab_size: int = 4096) -> None:
        """Fit BPE merges on training stories only."""
        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise ValueError(f"vocab_size must be greater than {len(SPECIAL_TOKENS)}.")
        if not stories:
            raise ValueError("Cannot train BPE on an empty story list.")

        cleaned = [story.strip() for story in stories if story and story.strip()]
        if not cleaned:
            raise ValueError("Training stories are empty after stripping.")

        # Byte-level BPE covers UTF-8 text with near-zero <unk> rate while still
        # exposing an explicit <unk> id for diagnostics and API stability.
        hf = HFTokenizer(BPE(unk_token="<unk>"))
        hf.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            show_progress=False,
            initial_alphabet=ByteLevel.alphabet(),
        )
        hf.train_from_iterator(cleaned, trainer=trainer)
        # Decode byte-level pieces back to UTF-8 text. Do not auto-insert BOS/EOS
        # here; encode(..., add_bos/add_eos) controls special tokens explicitly.
        hf.decoder = ByteLevelDecoder()
        self._tokenizer = hf
        self._configured_vocab_size = vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Return token ids. Special tokens are optional and off by default."""
        if self._tokenizer is None:
            raise RuntimeError("Call train() or load() before encode().")
        ids = list(self._tokenizer.encode(text).ids)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode ids to text. Special tokens are skipped for user-facing output."""
        if self._tokenizer is None:
            raise RuntimeError("Call train() or load() before decode().")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_pieces(self, text: str) -> list[str]:
        """Readable BPE pieces for examples and reports."""
        if self._tokenizer is None:
            raise RuntimeError("Call train() or load() before token_pieces().")
        return list(self._tokenizer.encode(text).tokens)

    def count_unk(self, token_ids: list[int]) -> int:
        unk = self.unk_token_id
        return sum(1 for token_id in token_ids if token_id == unk)

    def save(self, path: str) -> None:
        if self._tokenizer is None:
            raise RuntimeError("Nothing to save. Train or load a tokenizer first.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(destination))
        meta_path = destination.with_suffix(destination.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "tokenizer_type": "bpe",
                    "configured_vocab_size": self._configured_vocab_size,
                    "actual_vocab_size": self.vocab_size,
                    "special_tokens": list(SPECIAL_TOKENS),
                    "pad_token_id": self.pad_token_id,
                    "unk_token_id": self.unk_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"BPE tokenizer not found: {path}")
        hf = HFTokenizer.from_file(str(file_path))
        configured = None
        meta_path = file_path.with_suffix(file_path.suffix + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            configured = meta.get("configured_vocab_size")
        return cls(hf_tokenizer=hf, vocab_size=configured)
