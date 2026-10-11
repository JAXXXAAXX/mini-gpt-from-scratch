"""Token IDs -> token embedding + position embedding -> [B, T, C]."""

import torch
import torch.nn as nn


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
