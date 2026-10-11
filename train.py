"""Train MiniGPT to predict the next character in TinyStories."""

import sys

import torch
from torch.utils.data import DataLoader

import config
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import Tokenizer
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


def main():
    # Resume must reload data/processed/tokenizer.json. Do not fit a new vocabulary
    # onto a checkpoint, or the token ids will no longer match the saved weights.
    resume_path = config.RESUME_CHECKPOINT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    train_stories, valid_stories = load_tinystories(num_train=5000, num_valid=500)
    train_text = prepare_text(train_stories)
    valid_text = prepare_text(valid_stories)

    if resume_path:
        tokenizer = Tokenizer.load(config.TOKENIZER_PATH)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit(train_text)
        tokenizer.save(config.TOKENIZER_PATH)

    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
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
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        **trained,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
        "train_stories": len(train_stories),
        "valid_stories": len(valid_stories),
        "train_tokens": len(train_ids),
        "valid_tokens": len(valid_ids),
    }
    save_training_artifacts(result)
    print("Saved training history:")
    print(config.HISTORY_PATH)
    print("Saved training report:")
    print(config.REPORT_PATH)
    print("Saved loss curve:")
    print(config.LOSS_CURVE_PATH)


if __name__ == "__main__":
    main()
