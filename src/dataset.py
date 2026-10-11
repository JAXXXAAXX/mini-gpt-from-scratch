"""Token IDs -> Dataset -> (x, y) pairs for next-token prediction."""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Windows of token ids, each paired with the same window shifted one token ahead."""

    def __init__(self, token_ids: list[int], block_size: int):
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        # y starts one token later, so the sequence must be longer than one window.
        if len(token_ids) <= block_size:
            raise ValueError(
                "Need at least block_size + 1 tokens so every sample has a next-token target."
            )
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int):
        # x is the current tokens. y is that window shifted one token to the
        # right, so each position in x is trained to predict the next token.
        x = self.token_ids[index : index + self.block_size]
        y = self.token_ids[index + 1 : index + self.block_size + 1]
        return x, y
