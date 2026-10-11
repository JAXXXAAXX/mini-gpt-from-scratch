"""Train MiniGPT for next-token prediction on TinyStories.

Supports character (v1) and BPE (v2) tokenizers via config.TOKENIZER_TYPE.
BPE training never overwrites the v1 character checkpoint or tokenizer.
"""

import sys

import torch
from torch.utils.data import DataLoader

import config
from src.bpe_tokenizer import BPETokenizer
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import Tokenizer
from src.tokenizer_utils import encode_stories, parameter_breakdown, resolve_tokenizer_type
from src.trainer import (
    load_checkpoint,
    overfit_one_batch,
    save_training_artifacts,
    train_steps,
)


def build_model(vocab_size: int, device: torch.device) -> MiniGPT:
    return MiniGPT(
        vocab_size=vocab_size,
        block_size=config.BLOCK_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)


def build_or_load_tokenizer(train_stories: list[str], resume_path: str | None):
    kind = resolve_tokenizer_type()
    if kind == "bpe":
        if resume_path:
            tokenizer = BPETokenizer.load(config.BPE_TOKENIZER_PATH)
        else:
            tokenizer = BPETokenizer()
            print(f"Training BPE tokenizer on {len(train_stories)} train stories only...")
            tokenizer.train(train_stories, vocab_size=config.BPE_VOCAB_SIZE)
            tokenizer.save(config.BPE_TOKENIZER_PATH)
            print(f"Saved BPE tokenizer: {config.BPE_TOKENIZER_PATH}")
            print(f"BPE vocab size: {tokenizer.vocab_size}")
        return tokenizer, True

    train_text = prepare_text(train_stories)
    if resume_path:
        tokenizer = Tokenizer.load(config.CHAR_TOKENIZER_PATH)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit(train_text)
        tokenizer.save(config.CHAR_TOKENIZER_PATH)
    return tokenizer, False


def main():
    # Resume must reload the matching saved tokenizer. Do not fit a new vocabulary
    # onto a checkpoint, or the token ids will no longer match the saved weights.
    resume_path = config.RESUME_CHECKPOINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_type = resolve_tokenizer_type()
    print("Device:", device.type)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("Tokenizer type:", tokenizer_type)
    print("Block size:", config.BLOCK_SIZE, "tokenizer units")
    if tokenizer_type == "bpe":
        print(
            "Note: 64 BPE tokens span much more text than 64 character tokens. "
            "block_size is intentionally unchanged for this controlled comparison."
        )

    train_stories, valid_stories = load_tinystories(
        num_train=config.NUM_TRAIN_STORIES,
        num_valid=config.NUM_VALID_STORIES,
    )
    tokenizer, use_boundaries = build_or_load_tokenizer(train_stories, resume_path)

    train_ids = encode_stories(tokenizer, train_stories, use_story_boundaries=use_boundaries)
    valid_ids = encode_stories(tokenizer, valid_stories, use_story_boundaries=use_boundaries)
    train_chars = sum(len(story) for story in train_stories)
    characters_per_token = train_chars / len(train_ids) if train_ids else 0.0

    train_loader = DataLoader(
        TextDataset(train_ids, config.BLOCK_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TextDataset(valid_ids, config.BLOCK_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # Always initialize a fresh model for a new tokenizer vocabulary.
    model = build_model(tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    start_step = 0
    if resume_path:
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_step = checkpoint["step"]
        print(f"Resumed from {resume_path} at step {start_step}")

    if "--overfit" in sys.argv:
        # A fresh optimizer state on one repeated batch. This does not touch checkpoints.
        losses = overfit_one_batch(
            model,
            optimizer,
            next(iter(train_loader)),
            device,
            config.OVERFIT_STEPS,
            config.GRAD_CLIP,
        )
        print(f"Overfit steps: {config.OVERFIT_STEPS}")
        print(f"First loss: {losses[0]:.4f}")
        print(f"Last loss: {losses[-1]:.4f}")
        if losses[-1] >= losses[0]:
            raise RuntimeError("Overfit check failed: loss did not decrease on one repeated batch.")
        return

    trained = train_steps(
        model,
        optimizer,
        train_loader,
        valid_loader,
        device,
        start_step=start_step,
        best_checkpoint_path=config.BEST_CHECKPOINT,
        latest_checkpoint_path=config.LATEST_CHECKPOINT,
        characters_per_token=characters_per_token,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    params = parameter_breakdown(tokenizer.vocab_size)
    result = {
        **trained,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": config.TOKENIZER_PATH,
        "vocab_size": tokenizer.vocab_size,
        "block_size": config.BLOCK_SIZE,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "max_steps": config.MAX_STEPS,
        "eval_interval": config.EVAL_INTERVAL,
        "eval_batches": config.EVAL_BATCHES,
        "grad_clip": config.GRAD_CLIP,
        "parameter_count": parameter_count,
        "parameter_breakdown": params,
        "train_stories": len(train_stories),
        "valid_stories": len(valid_stories),
        "train_tokens": len(train_ids),
        "valid_tokens": len(valid_ids),
        "train_characters": train_chars,
        "story_boundaries": use_boundaries,
    }
    save_training_artifacts(result)
    print("Saved training history:")
    print(config.HISTORY_PATH)
    print("Saved training report:")
    print(config.REPORT_PATH)
    print("Saved loss curve:")
    print(config.LOSS_CURVE_PATH)
    print("Best checkpoint:")
    print(config.BEST_CHECKPOINT)


if __name__ == "__main__":
    main()
