"""Select and prepare character or BPE tokenizers for MiniGPT experiments."""

from __future__ import annotations

from typing import Any

import config
from src.bpe_tokenizer import BPETokenizer
from src.data_utils import prepare_text
from src.model import MiniGPT
from src.tokenizer import Tokenizer


def resolve_tokenizer_type(tokenizer_type: str | None = None) -> str:
    value = (tokenizer_type or config.TOKENIZER_TYPE).lower()
    if value not in {"char", "bpe"}:
        raise ValueError(f"TOKENIZER_TYPE must be 'char' or 'bpe', got {value!r}.")
    return value


def tokenizer_path_for(tokenizer_type: str | None = None) -> str:
    kind = resolve_tokenizer_type(tokenizer_type)
    return config.BPE_TOKENIZER_PATH if kind == "bpe" else config.CHAR_TOKENIZER_PATH


def load_tokenizer(tokenizer_type: str | None = None) -> Tokenizer | BPETokenizer:
    kind = resolve_tokenizer_type(tokenizer_type)
    path = tokenizer_path_for(kind)
    if kind == "bpe":
        return BPETokenizer.load(path)
    return Tokenizer.load(path)


def encode_stories(
    tokenizer: Tokenizer | BPETokenizer,
    stories: list[str],
    *,
    use_story_boundaries: bool,
) -> list[int]:
    """Encode stories into one continuous token stream for TextDataset."""
    if use_story_boundaries:
        return encode_documents(tokenizer, stories)

    return tokenizer.encode(prepare_text(stories))


def encode_documents(tokenizer: Tokenizer | BPETokenizer, documents: list[str]) -> list[int]:
    """Encode each document with BOS/EOS, then concatenate for windowed training."""
    ids: list[int] = []
    for document in documents:
        text = document.strip()
        if not text:
            continue
        if isinstance(tokenizer, BPETokenizer):
            ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        else:
            ids.extend(tokenizer.encode(text))
    if not ids:
        raise ValueError("Encoding documents produced an empty token stream.")
    return ids


def story_token_counts(
    tokenizer: Tokenizer | BPETokenizer,
    stories: list[str],
    *,
    use_story_boundaries: bool,
) -> list[int]:
    counts = []
    for story in stories:
        text = story.strip()
        if not text:
            continue
        if use_story_boundaries and isinstance(tokenizer, BPETokenizer):
            counts.append(len(tokenizer.encode(text, add_bos=True, add_eos=True)))
        else:
            counts.append(len(tokenizer.encode(text)))
    return counts


def parameter_breakdown(
    vocab_size: int,
    block_size: int = config.BLOCK_SIZE,
    embedding_dim: int = config.EMBEDDING_DIM,
    num_heads: int = config.NUM_HEADS,
    num_layers: int = config.NUM_LAYERS,
    dropout: float = config.DROPOUT,
    position_encoding: str = config.POSITION_ENCODING,
) -> dict[str, Any]:
    """Instantiate MiniGPT on CPU and break out major parameter groups."""
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        position_encoding=position_encoding,
    )
    token_embedding = sum(p.numel() for p in model.token_embedding.parameters())
    if model.position_embedding is None:
        position_embedding = 0
    else:
        position_embedding = sum(p.numel() for p in model.position_embedding.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    transformer = sum(p.numel() for p in model.blocks.parameters()) + sum(
        p.numel() for p in model.final_ln.parameters()
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "vocab_size": vocab_size,
        "token_embedding_params": token_embedding,
        "position_embedding_params": position_embedding,
        "transformer_params": transformer,
        "lm_head_params": lm_head,
        "total_params": total,
    }


def eos_token_id(tokenizer: Tokenizer | BPETokenizer) -> int | None:
    return getattr(tokenizer, "eos_token_id", None)


def has_eos(tokenizer: Tokenizer | BPETokenizer) -> bool:
    return eos_token_id(tokenizer) is not None
