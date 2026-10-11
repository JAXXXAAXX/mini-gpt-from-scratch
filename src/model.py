"""Token IDs -> embeddings -> one causal self-attention head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    """Input representation only. Attention comes later."""

    def __init__(self, vocab_size: int, block_size: int, embedding_dim: int):
        super().__init__()
        self.block_size = block_size
        # Token IDs are integer indexes, not semantic vectors.
        # This table maps each id to a learnable vector of length embedding_dim.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Each position 0 .. block_size-1 gets its own learnable vector.
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x holds token ids with shape [B, T].
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(
                f"Batch of {B} sequences has length {T}, which is longer than block_size {self.block_size}."
            )

        # Positions are 0, 1, 2, ..., T - 1, on the same device as the token ids.
        positions = torch.arange(T, device=x.device)
        token_emb = self.token_embedding(x)  # [B, T] -> [B, T, C]
        position_emb = self.position_embedding(positions)  # [T] -> [T, C]
        # Both tensors use embedding_dim, so [T, C] broadcasts across the
        # batch of size B and adds position information to every token vector.
        return token_emb + position_emb  # [B, T, C]


class SelfAttentionHead(nn.Module):
    """One causal self-attention head.

    Q is what the current token is looking for. K is how each token can be
    matched. Their dot product is a compatibility score. Softmax turns those
    scores into weights, and those weights mix the V vectors, which hold the
    information that is actually aggregated. The causal mask stops a token
    from reading future tokens.
    """

    def __init__(self, embedding_dim: int, head_dim: int, block_size: int, dropout: float):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        # A buffer moves with .to(device), unlike a plain tensor saved on self.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # x is the input representation, shape [B, T, C].
        batch_size, seq_len, channels = x.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} is longer than block_size {self.block_size}."
            )
        if channels != self.query.in_features:
            raise ValueError(
                f"Expected embedding dim {self.query.in_features}, got {channels}."
            )

        # Each linear layer maps [B, T, C] to [B, T, H].
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # scores[b, i, j] is how well token i's query matches token j's key.
        # K is transposed so each query dots with every key: [B, T, H] @ [B, H, T].
        scores = q @ k.transpose(-2, -1)
        # Dot products grow with head_dim. Dividing by sqrt(H) keeps softmax stable.
        scores = scores / math.sqrt(self.head_dim)

        # True means "allowed". Future positions j > i are set to -inf so softmax
        # gives them probability 0.
        allowed = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~allowed, float("-inf"))
        # Softmax over the last dimension normalizes across keys for each query.
        weights = F.softmax(scores, dim=-1)
        mixed = self.attn_dropout(weights) @ v  # [B, T, T] @ [B, T, H] -> [B, T, H]

        if not return_details:
            return mixed
        return mixed, {
            "q": q,
            "k": k,
            "v": v,
            "attention_scores": scores,
            "attention_weights": weights,
            "output": mixed,
            "batch_size": batch_size,
        }
