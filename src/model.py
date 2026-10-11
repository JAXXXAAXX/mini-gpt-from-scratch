"""Token IDs -> stacked Transformer blocks -> vocabulary logits."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    """Token ids to next-token logits.

    Embeddings produce [B, T, C]. Each Transformer block keeps that shape.
    The LM head then maps every position to one raw score per vocabulary token.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Token IDs are integer indexes, not semantic vectors.
        # This table maps each id to a learnable vector of length embedding_dim.
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Each position 0 .. block_size-1 gets its own learnable vector.
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, block_size, dropout)
                for _ in range(num_layers)
            ]
        )
        # Normalize the hidden states before they are scored against the vocabulary.
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # x holds token ids with shape [B, T].
        batch_size, seq_len = x.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Batch of {batch_size} sequences has length {seq_len}, "
                f"which is longer than block_size {self.block_size}."
            )

        # Positions are 0, 1, 2, ..., T - 1, on the same device as the token ids.
        positions = torch.arange(seq_len, device=x.device)
        token_emb = self.token_embedding(x)  # [B, T] -> [B, T, C]
        position_emb = self.position_embedding(positions)  # [T] -> [T, C]
        # Both tensors use embedding_dim, so [T, C] broadcasts across the batch.
        hidden = token_emb + position_emb  # [B, T, C]

        block_outputs = []
        for block in self.blocks:
            hidden = block(hidden)
            if return_details:
                block_outputs.append(hidden)
        normalized = self.final_ln(hidden)
        # logits[b, t, :] is one raw score for every vocabulary token at that position.
        # logits[0, 10, :] is the next-token scores for sequence 0 at position 10.
        # Softmax is not applied here. CrossEntropyLoss combines log-softmax and
        # negative log likelihood, and it is more stable when given these raw scores.
        logits = self.lm_head(normalized)  # [B, T, C] -> [B, T, vocab_size]
        if not return_details:
            return logits
        return logits, {
            "token_ids": x,
            "embeddings": token_emb + position_emb,
            "blocks": block_outputs,
            "final_ln": normalized,
            "logits": logits,
        }

    def architecture_summary(self) -> str:
        return (
            "MiniGPT\n"
            "├── Token Embedding\n"
            "├── Position Embedding\n"
            f"├── Transformer Blocks × {self.num_layers}\n"
            "├── Final LayerNorm\n"
            "└── LM Head\n"
            "\n"
            f"Embedding Dim: {self.embedding_dim}\n"
            f"Heads: {self.num_heads}\n"
            f"Layers: {self.num_layers}\n"
            f"Context Length: {self.block_size}\n"
            f"Vocabulary Size: {self.vocab_size}"
        )

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Sample new tokens one at a time and append them to the context.

        Softmax is used only here. Training keeps raw logits for CrossEntropyLoss.
        When eos_token_id is set, sampling stops early after that token is produced.
        """
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape [B, T], got {tuple(token_ids.shape)}.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be greater than 0.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be None or greater than 0.")
        if eos_token_id is not None and not (0 <= eos_token_id < self.vocab_size):
            raise ValueError("eos_token_id must be None or a valid vocabulary index.")

        was_training = self.training
        self.eval()
        generated = token_ids
        for _ in range(max_new_tokens):
            # The model was trained with at most block_size tokens of context.
            context = generated[:, -self.block_size :]
            logits = self(context)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Generation produced NaN or infinite logits.")
            # Only the last position predicts the token that comes next.
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                keep = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, keep)
                cutoff = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            if not ((next_token >= 0) & (next_token < self.vocab_size)).all():
                raise RuntimeError("Sampled a token id outside the vocabulary.")
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        if was_training:
            self.train()
        return generated


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


class MultiHeadAttention(nn.Module):
    """Several causal heads, concatenated, then mixed by an output projection."""

    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        # Each head has its own Q, K, and V, so each can learn a different pattern.
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(embedding_dim, self.head_dim, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # Every head sees the same [B, T, C] input and returns [B, T, head_dim].
        head_outputs = []
        weight_matrices = []
        for head in self.heads:
            if return_details:
                head_output, details = head(x, return_details=True)
                weight_matrices.append(details["attention_weights"])
            else:
                head_output = head(x)
            head_outputs.append(head_output)

        # Concatenate on the channel axis: 4 x [B, T, 32] -> [B, T, 128].
        # Different heads can learn different token relationships. Their outputs
        # are joined so the next layer sees all of those views together.
        concatenated = torch.cat(head_outputs, dim=-1)
        projected = self.output_projection(concatenated)
        output = self.output_dropout(projected)
        if not return_details:
            return output
        return output, {
            "head_outputs": head_outputs,
            "concatenated": concatenated,
            "projected": projected,
            "output": output,
            "attention_weights": weight_matrices,
        }


class FeedForward(nn.Module):
    """Per-token nonlinear map. It does not mix positions in the sequence.

    Attention gathers information across tokens. This network then transforms
    each token vector on its own: the linear layers act only on the last dimension.
    """

    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.expand = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.project = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C -> 4C gives each token a wider intermediate feature space, then back to C.
        hidden = self.activation(self.expand(x))
        return self.dropout(self.project(hidden))


class TransformerBlock(nn.Module):
    """One Pre-Norm block: attention, then a feed-forward network, each with a residual."""

    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # LayerNorm stabilizes the representation before the sublayer.
        normed = self.ln1(x)
        # Attention is token-to-token communication: each token reads earlier tokens.
        attended = self.attention(normed)
        # The residual keeps the original x and gives gradients a direct path backward.
        residual_1 = x + attended

        normed_2 = self.ln2(residual_1)
        # The FFN is a per-token nonlinear transform. It does not mix sequence positions.
        transformed = self.ffn(normed_2)
        output = residual_1 + transformed

        if output.shape != x.shape:
            raise RuntimeError(f"Transformer block changed shape from {tuple(x.shape)} to {tuple(output.shape)}.")
        if not return_details:
            return output
        return output, {
            "input": x,
            "layernorm_1": normed,
            "attention": attended,
            "residual_1": residual_1,
            "layernorm_2": normed_2,
            "ffn": transformed,
            "output": output,
        }
