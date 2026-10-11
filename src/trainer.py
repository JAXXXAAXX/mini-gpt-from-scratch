"""Next-token training, validation, checkpoints, and the training report."""

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from src.model import MiniGPT


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Score every position in the batch against its next-token target.

    logits are [B, T, V] and targets are [B, T]. Flattening makes B * T
    separate predictions. A batch of 32 sequences of length 64 is 2048 targets.
    CrossEntropyLoss wants raw logits. It applies log-softmax internally.
    """
    batch_size, seq_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
    )


def perplexity(loss: float) -> float:
    """exp(loss). Very large losses are reported as infinity instead of overflowing."""
    if loss > 20:
        return float("inf")
    return math.exp(loss)


def _finite(loss: torch.Tensor, step: int, split: str) -> float:
    value = float(loss.detach())
    if not math.isfinite(value):
        raise RuntimeError(f"{split} loss became non-finite at step {step}: {value}")
    return value


def evaluate(model: MiniGPT, loader, device: torch.device, max_batches: int, step: int) -> float:
    """Average next-token loss. No backward pass and no optimizer update."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            total += _finite(language_model_loss(model(inputs), targets), step, "Validation")
            count += 1
    if was_training:
        model.train()
    if count == 0:
        raise RuntimeError("The loader produced no batches to evaluate.")
    return total / count


def train_one_step(model, optimizer, inputs, targets, grad_clip: float, step: int) -> float:
    """One AdamW update on a single next-token batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = language_model_loss(model(inputs), targets)
    value = _finite(loss, step, "Train")
    loss.backward()
    # Clip the global gradient norm so one spike cannot take a huge step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def overfit_one_batch(model, optimizer, batch, device: torch.device, steps: int, grad_clip: float) -> list[float]:
    """Repeat one batch to check that loss, backward, and AdamW are connected.

    This is a wiring check. A drop here is not evidence of TinyStories quality.
    """
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.eval()
    with torch.no_grad():
        initial = float(language_model_loss(model(inputs), targets))
    model.train()
    losses = [initial]
    for step in range(1, steps + 1):
        losses.append(train_one_step(model, optimizer, inputs, targets, grad_clip, step))
    return losses


def save_checkpoint(path: Path, model, optimizer, step: int, train_loss: float, validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_config": {
                "vocab_size": model.vocab_size,
                "block_size": model.block_size,
                "embedding_dim": model.embedding_dim,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "dropout": config.DROPOUT,
            },
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_steps(
    model,
    optimizer,
    train_loader,
    valid_loader,
    device: torch.device,
    start_step: int = 0,
    best_validation_loss: float = float("inf"),
) -> dict:
    """Train for config.MAX_STEPS and record loss at each eval interval."""
    model.train()
    history = []
    train_iterator = iter(train_loader)
    train_seconds = 0.0
    train_tokens = 0
    best_loss = best_validation_loss
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record(step: int, train_loss: float, validation_loss: float) -> None:
        nonlocal best_loss
        elapsed = time.perf_counter() - started
        history.append(
            {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        print(f"Step {step}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print()
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            step,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                step,
                train_loss,
                validation_loss,
            )

    if start_step == 0:
        initial_inputs, initial_targets = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            initial_loss = _finite(
                language_model_loss(model(initial_inputs.to(device)), initial_targets.to(device)),
                0,
                "Train",
            )
        model.train()
        record(0, initial_loss, evaluate(model, valid_loader, device, config.EVAL_BATCHES, 0))

    recent_losses = []
    for step in range(start_step + 1, config.MAX_STEPS + 1):
        (inputs, targets), train_iterator = next_batch(train_loader, train_iterator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _sync(device)
        step_started = time.perf_counter()
        recent_losses.append(train_one_step(model, optimizer, inputs, targets, config.GRAD_CLIP, step))
        _sync(device)
        train_seconds += time.perf_counter() - step_started
        train_tokens += inputs.shape[0] * inputs.shape[1]

        if step % config.EVAL_INTERVAL == 0 or step == config.MAX_STEPS:
            train_loss = sum(recent_losses) / len(recent_losses)
            recent_losses.clear()
            record(step, train_loss, evaluate(model, valid_loader, device, config.EVAL_BATCHES, step))

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    throughput = train_tokens / train_seconds if train_seconds > 0 else 0.0
    best_entry = min(history, key=lambda item: item["validation_loss"])
    return {
        "history": history,
        "tokens_per_second": throughput,
        "peak_gpu_memory_mb": peak_mb,
        "best_step": best_entry["step"],
        "best_validation_loss": best_entry["validation_loss"],
        "best_validation_perplexity": perplexity(best_entry["validation_loss"]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_loss_curve(history: list[dict], path: Path) -> None:
    steps = [item["step"] for item in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [item["train_loss"] for item in history], label="train")
    plt.plot(steps, [item["validation_loss"] for item in history], label="validation")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def generate_training_report(result: dict) -> str:
    """Build the Markdown report from the saved training result dictionary."""
    history = result["history"]
    rows = "\n".join(
        f"| {item['step']} | {item['train_loss']:.4f} | {item['validation_loss']:.4f} | "
        f"{item['elapsed_seconds']:.1f} s |"
        for item in history
    )
    first = history[0]
    last = history[-1]
    memory = (
        "unavailable"
        if result["peak_gpu_memory_mb"] is None
        else f"{result['peak_gpu_memory_mb']:.2f} MB"
    )
    gpu_name = result["gpu_name"] or "none"
    if last["validation_loss"] < first["validation_loss"]:
        observation = (
            f"Validation loss fell from {first['validation_loss']:.4f} at step {first['step']} "
            f"to {last['validation_loss']:.4f} at step {last['step']}."
        )
    else:
        observation = (
            f"Validation loss did not fall. It moved from {first['validation_loss']:.4f} "
            f"to {last['validation_loss']:.4f}."
        )
    return f"""# MiniGPT Training Report

## Configuration

- Device: `{result["device"]}`
- GPU name: {gpu_name}
- Vocabulary size: {result["vocab_size"]}
- Block size: {result["block_size"]}
- Embedding dimension: {result["embedding_dim"]}
- Heads: {result["num_heads"]}
- Layers: {result["num_layers"]}
- Dropout: {result["dropout"]}
- Batch size: {result["batch_size"]}
- Learning rate: {result["learning_rate"]}
- Max steps: {result["max_steps"]}
- Eval interval: {result["eval_interval"]}
- Eval batches: {result["eval_batches"]}
- Gradient clip: {result["grad_clip"]}

## Model Size

{result["parameter_count"]:,} trainable parameters ({result["parameter_count"] / 1_000_000:.2f}M).

## Dataset

- Training stories: {result["train_stories"]:,}
- Validation stories: {result["valid_stories"]:,}
- Training tokens: {result["train_tokens"]:,}
- Validation tokens: {result["valid_tokens"]:,}
- Targets per batch: {result["batch_size"] * result["block_size"]:,}

## Training Setup

AdamW updates the model on next-token cross-entropy. Each logged train loss is the average of the training steps since the previous log. Validation loss is the average over {result["eval_batches"]} validation batches, with gradients disabled. Checkpoints are `checkpoints/latest.pt` and `checkpoints/best.pt`.

## Loss Progress

![Loss curve](loss_curve.png)

| Step | Train Loss | Validation Loss | Elapsed Time |
|---:|---:|---:|---:|
{rows}

## Best Validation Result

- Step: {result["best_step"]}
- Validation loss: {result["best_validation_loss"]:.4f}
- Validation perplexity: {result["best_validation_perplexity"]:.2f}

Perplexity is `exp(validation loss)`. A lower value means the model assigns more probability to the actual next characters. It is not a complete measure of story quality.

## Computational Performance

- Training throughput: {result["tokens_per_second"]:,.0f} tokens/sec
- Peak GPU memory: {memory}
- Elapsed time, including validation: {result["elapsed_seconds"]:.1f} s

Throughput counts only optimizer steps. Validation time is excluded.

## Observations

{observation} Training loss moved from {first["train_loss"]:.4f} to {last["train_loss"]:.4f}. These curves show that the training pipeline runs. They do not show that the model can write stories yet.

## Limitations

- The run uses a 5,000-story TinyStories slice and a character vocabulary.
- There is no text generation yet.
- Perplexity is only a summary of next-character loss.
- A short run can still leave the validation loss far from a useful language model.

## Next Step

Generate text from `checkpoints/best.pt` with the saved tokenizer, without fitting a new vocabulary.
"""


def save_training_artifacts(result: dict) -> None:
    history_path = Path(config.HISTORY_PATH)
    report_path = Path(config.REPORT_PATH)
    curve_path = Path(config.LOSS_CURVE_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    save_loss_curve(result["history"], curve_path)
    history_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(generate_training_report(result), encoding="utf-8")
