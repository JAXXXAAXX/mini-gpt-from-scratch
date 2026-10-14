"""Train MiniGPT for next-token prediction on TinyStories.

Supports v1 through v2.7 via config.EXPERIMENT.
BPE training never overwrites an existing BPE vocabulary.
v2.2 resumes v2.1 and writes only to the v2.2 checkpoint paths.
v2.4 trains a fresh RoPE model on 20,000 stories and does not overwrite v2.3.
v2.5 widens that RoPE model and does not overwrite v2.4.
v2.6/v2.7 fine-tune v2.5 on User/Assistant dialogue for topical replies.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from src.bpe_tokenizer import BPETokenizer
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.dialogue_data import (
    dialogue_and_story_documents,
    load_daily_dialogues,
)
from src.model import MiniGPT
from src.tokenizer import Tokenizer
from src.tokenizer_utils import (
    encode_documents,
    encode_stories,
    parameter_breakdown,
    resolve_tokenizer_type,
)
from src.trainer import (
    load_checkpoint,
    overfit_one_batch,
    save_checkpoint,
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
        position_encoding=config.POSITION_ENCODING,
    ).to(device)


def build_or_load_tokenizer(train_stories: list[str], resume_path: str | None):
    kind = resolve_tokenizer_type()
    if kind == "bpe":
        bpe_path = Path(config.BPE_TOKENIZER_PATH)
        if bpe_path.exists():
            # Keep the v2 vocabulary. A new merge file would make v2 and v3 incomparable.
            tokenizer = BPETokenizer.load(config.BPE_TOKENIZER_PATH)
            print(f"Loaded existing BPE tokenizer: {config.BPE_TOKENIZER_PATH}")
            print(f"BPE vocab size: {tokenizer.vocab_size}")
            return tokenizer, True
        if resume_path:
            raise FileNotFoundError(
                f"Cannot resume a BPE run without the tokenizer at {config.BPE_TOKENIZER_PATH}."
            )
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
    print("Experiment:", config.EXPERIMENT)
    print("Tokenizer type:", tokenizer_type)
    print("Block size:", config.BLOCK_SIZE, "tokenizer units")
    print("Batch size:", config.BATCH_SIZE)
    print("Tokens per step:", config.BATCH_SIZE * config.BLOCK_SIZE)
    if config.EXPERIMENT == "v2_8_daily_dialog":
        print(
            "v2.8 fine-tunes from v2.7 on real DailyDialog only. "
            "No hand-written reply templates are added. "
            f"Learning rate {config.LEARNING_RATE}, target step {config.MAX_STEPS}."
        )
    elif config.EXPERIMENT == "v2_7_reply_generalize":
        print(
            "v2.7 checkpoint is inference-only here; hand-written reply templates "
            "were deleted. For new chat fine-tuning use EXPERIMENT=v2_8_daily_dialog."
        )
    elif config.EXPERIMENT == "v2_6_reply_ft":
        print(
            "v2.6 checkpoint is legacy; hand-written reply templates were deleted. "
            "For new chat fine-tuning use EXPERIMENT=v2_8_daily_dialog."
        )
    elif config.EXPERIMENT == "v2_5_width256":
        print(
            "v2.5 keeps RoPE, context 256, batch 8, and the 20,000-story slice. "
            f"Width changes to embedding {config.EMBEDDING_DIM}, "
            f"{config.NUM_HEADS} heads, {config.NUM_LAYERS} layers. "
            f"It trains from fresh weights for {config.MAX_STEPS} steps and does not resume v2.4."
        )
    elif config.EXPERIMENT == "v2_4_stories20k":
        print(
            "v2.4 keeps the v2.3 RoPE model, context 256, and batch 8. "
            f"It trains from fresh weights on {config.NUM_TRAIN_STORIES} stories "
            f"for {config.MAX_STEPS} steps. Validation stays "
            f"{config.NUM_VALID_STORIES} stories. The existing BPE file is not refit."
        )
    elif config.EXPERIMENT == "v2_3_rope_ctx256":
        print(
            "v2.3 trains a fresh RoPE model for the same 3000-step budget as v2.2. "
            "It does not resume v2.2."
        )
    elif config.EXPERIMENT == "v2_2_ctx256_longtrain":
        print(
            "v2.2 keeps context 256 and batch 8 (2048 tokens/step) and trains "
            f"through {config.MAX_STEPS} total steps."
        )
    elif config.EXPERIMENT == "v2_1_ctx256":
        print(
            "v2.1 changes only context length. Batch size is 8 so each step still "
            "sees 2048 tokens, matching v2 (batch 32 x context 64)."
        )
    elif tokenizer_type == "bpe":
        print(
            "Note: 64 BPE tokens span much more text than 64 character tokens. "
            "block_size is intentionally unchanged for this controlled comparison."
        )

    train_stories, valid_stories = load_tinystories(
        num_train=config.NUM_TRAIN_STORIES,
        num_valid=config.NUM_VALID_STORIES,
    )
    tokenizer, use_boundaries = build_or_load_tokenizer(train_stories, resume_path)

    dialogue_train_docs: list[str] = []
    dialogue_valid_docs: list[str] = []
    if config.DATA_MODE == "daily_dialog":
        dialogue_train_docs, dialogue_valid_docs = load_daily_dialogues(
            num_train=config.DAILY_DIALOG_TRAIN,
            num_valid=config.DAILY_DIALOG_VALID,
        )
        train_docs = dialogue_and_story_documents(
            dialogue_train_docs,
            train_stories,
            story_ratio=config.STORY_MIX_RATIO,
        )
        train_ids = encode_documents(tokenizer, train_docs)
        valid_ids = encode_documents(tokenizer, dialogue_valid_docs)
        train_chars = sum(len(doc) for doc in train_docs)
        print(
            f"DailyDialog train docs: {len(dialogue_train_docs)}; "
            f"mixed train docs: {len(train_docs)}; "
            f"DailyDialog valid docs: {len(dialogue_valid_docs)}"
        )
    elif config.DATA_MODE == "dialogue_mix":
        raise ValueError(
            "DATA_MODE 'dialogue_mix' was removed with the hand-written reply templates. "
            "Use DATA_MODE 'daily_dialog' for chat fine-tuning."
        )
    else:
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
    best_validation_loss = float("inf")
    initial_history = None
    resume_meta = {
        "resumed": False,
        "mode": "trained fresh",
        "position_encoding": config.POSITION_ENCODING,
    }
    if config.POSITION_ENCODING == "rope" and resume_path and not config.ALLOW_ROPE_RESUME:
        raise RuntimeError(
            "A RoPE run must start from fresh weights. "
            "RESUME_CHECKPOINT has to stay empty for this experiment."
        )
    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(
                f"Cannot resume because {resume_path} is missing. "
                "Refusing to train a fresh run in its place."
            )
        if config.EXPERIMENT in {"v2_6_reply_ft", "v2_7_reply_generalize", "v2_8_daily_dialog"}:
            # Keep v2.5 weights, but use a fresh optimizer at the fine-tune learning rate.
            # Dialogue validation is a new domain, so do not carry the TinyStories best loss.
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
            start_step = int(checkpoint["step"])
            resumed_train = float(checkpoint["train_loss"])
            resumed_val = float(checkpoint["validation_loss"])
            best_validation_loss = float("inf")
            resume_mode = f"fine-tuned from v2.5 on dialogue ({config.EXPERIMENT})"
            resume_note = (
                "Weights come from v2.5. The optimizer is restarted at the fine-tune "
                "learning rate. Validation now tracks dialogue loss, not TinyStories loss."
            )
        else:
            checkpoint = load_checkpoint(resume_path, model, optimizer, device)
            start_step = int(checkpoint["step"])
            resumed_train = float(checkpoint["train_loss"])
            resumed_val = float(checkpoint["validation_loss"])
            best_validation_loss = resumed_val
            resume_mode = "resumed from v2.1"
            resume_note = (
                "The target is 3000 total steps. This is not 1000 previous steps "
                "plus 3000 new steps."
            )
        if start_step >= config.MAX_STEPS:
            raise RuntimeError(
                f"Checkpoint step {start_step} is already at or past MAX_STEPS "
                f"{config.MAX_STEPS}. Refusing to train extra unplanned steps."
            )
        initial_history = [
            {
                "step": start_step,
                "train_loss": resumed_train,
                "validation_loss": resumed_val,
                "validation_perplexity": float(torch.exp(torch.tensor(resumed_val))),
                "mean_grad_norm": None,
                "max_grad_norm": None,
                "elapsed_seconds": 0.0,
                "source": "resumed_checkpoint",
            }
        ]
        resume_meta = {
            "resumed": True,
            "mode": resume_mode,
            "from_checkpoint": resume_path,
            "from_step": start_step,
            "target_total_steps": config.MAX_STEPS,
            "additional_steps": config.MAX_STEPS - start_step,
            "resumed_validation_loss": resumed_val,
            "note": resume_note,
        }
        print(
            f"Resumed from {resume_path} at step {start_step}. "
            f"Training continues through step {config.MAX_STEPS}."
        )

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

    if initial_history is not None and config.EXPERIMENT not in {
        "v2_6_reply_ft",
        "v2_7_reply_generalize",
        "v2_8_daily_dialog",
    }:
        # Keep the resumed step eligible as the best v2.2 model if later evals are worse.
        resumed_point = initial_history[0]
        save_checkpoint(
            Path(config.BEST_CHECKPOINT),
            model,
            optimizer,
            int(resumed_point["step"]),
            float(resumed_point["train_loss"]),
            float(resumed_point["validation_loss"]),
        )

    trained = train_steps(
        model,
        optimizer,
        train_loader,
        valid_loader,
        device,
        start_step=start_step,
        best_validation_loss=best_validation_loss,
        initial_history=initial_history,
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
        "experiment": config.EXPERIMENT,
        "resume": resume_meta,
        "loss_curve_path": config.LOSS_CURVE_PATH,
        "tokens_per_step": config.BATCH_SIZE * config.BLOCK_SIZE,
        "planned_target_tokens": config.BATCH_SIZE * config.BLOCK_SIZE * config.MAX_STEPS,
        "tokenizer_type": tokenizer_type,
        "tokenizer_path": config.TOKENIZER_PATH,
        "vocab_size": tokenizer.vocab_size,
        "block_size": config.BLOCK_SIZE,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "position_encoding": config.POSITION_ENCODING,
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
