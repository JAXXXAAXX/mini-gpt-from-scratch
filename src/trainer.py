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


def train_one_step(model, optimizer, inputs, targets, grad_clip: float, step: int) -> tuple[float, float]:
    """One AdamW update. Returns the loss and the pre-clip gradient norm."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = language_model_loss(model(inputs), targets)
    value = _finite(loss, step, "Train")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {name} at step {step}.")
    # Clip the global gradient norm so one spike cannot take a huge step.
    # The returned value is the total norm before clipping.
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"Gradient norm became non-finite at step {step}: {grad_norm}")
    optimizer.step()
    return value, grad_norm


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
        losses.append(train_one_step(model, optimizer, inputs, targets, grad_clip, step)[0])
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
                "version": config.EXPERIMENT,
                "experiment": config.EXPERIMENT,
                "tokenizer_type": config.TOKENIZER_TYPE,
                "position_encoding": getattr(model, "position_encoding", "absolute"),
                "vocab_size": model.vocab_size,
                "block_size": model.block_size,
                "batch_size": config.BATCH_SIZE,
                "embedding_dim": model.embedding_dim,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "dropout": config.DROPOUT,
                "learning_rate": config.LEARNING_RATE,
                "training_steps": config.MAX_STEPS,
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
    best_checkpoint_path: str | None = None,
    latest_checkpoint_path: str | None = None,
    characters_per_token: float | None = None,
    initial_history: list[dict] | None = None,
) -> dict:
    """Train for config.MAX_STEPS and record loss at each eval interval."""
    model.train()
    history = list(initial_history or [])
    train_iterator = iter(train_loader)
    train_seconds = 0.0
    train_tokens = 0
    best_loss = best_validation_loss
    best_path = Path(best_checkpoint_path or config.BEST_CHECKPOINT)
    latest_path = Path(latest_checkpoint_path or config.LATEST_CHECKPOINT)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record(
        step: int,
        train_loss: float,
        validation_loss: float,
        mean_grad_norm: float | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        nonlocal best_loss
        elapsed = time.perf_counter() - started
        history.append(
            {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity(validation_loss),
                "mean_grad_norm": mean_grad_norm,
                "max_grad_norm": max_grad_norm,
                "elapsed_seconds": elapsed,
            }
        )
        grad_text = "n/a" if mean_grad_norm is None else f"{mean_grad_norm:.4f}"
        print(f"Step {step}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Mean Grad Norm: {grad_text}")
        print()
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            step,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                best_path,
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
    recent_grads = []
    for step in range(start_step + 1, config.MAX_STEPS + 1):
        (inputs, targets), train_iterator = next_batch(train_loader, train_iterator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _sync(device)
        step_started = time.perf_counter()
        step_loss, step_grad = train_one_step(
            model, optimizer, inputs, targets, config.GRAD_CLIP, step
        )
        recent_losses.append(step_loss)
        recent_grads.append(step_grad)
        _sync(device)
        train_seconds += time.perf_counter() - step_started
        train_tokens += inputs.shape[0] * inputs.shape[1]

        if step % config.EVAL_INTERVAL == 0 or step == config.MAX_STEPS:
            train_loss = sum(recent_losses) / len(recent_losses)
            mean_grad = sum(recent_grads) / len(recent_grads)
            max_grad = max(recent_grads)
            recent_losses.clear()
            recent_grads.clear()
            record(
                step,
                train_loss,
                evaluate(model, valid_loader, device, config.EVAL_BATCHES, step),
                mean_grad,
                max_grad,
            )

    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    throughput = train_tokens / train_seconds if train_seconds > 0 else 0.0
    characters_per_second = None
    if characters_per_token is not None:
        characters_per_second = throughput * characters_per_token
    best_entry = min(history, key=lambda item: item["validation_loss"])
    return {
        "history": history,
        "tokens_per_second": throughput,
        "characters_per_second": characters_per_second,
        "characters_per_token": characters_per_token,
        "peak_gpu_memory_mb": peak_mb,
        "best_step": best_entry["step"],
        "best_validation_loss": best_entry["validation_loss"],
        "best_validation_perplexity": perplexity(best_entry["validation_loss"]),
        "elapsed_seconds": time.perf_counter() - started,
        "best_checkpoint_path": best_path.as_posix(),
        "latest_checkpoint_path": latest_path.as_posix(),
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
    def grad_cell(item: dict) -> str:
        value = item.get("mean_grad_norm")
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    rows = "\n".join(
        f"| {item['step']} | {item['train_loss']:.4f} | {item['validation_loss']:.4f} | "
        f"{perplexity(item['validation_loss']):.2f} | {grad_cell(item)} | "
        f"{item['elapsed_seconds']:.1f} s |"
        for item in history
    )
    curve_name = Path(result.get("loss_curve_path") or config.LOSS_CURVE_PATH).name
    tokens_per_step = result["batch_size"] * result["block_size"]
    exposure = tokens_per_step * result["max_steps"]
    resume = result.get("resume") or {}
    if resume.get("resumed"):
        resume_section = f"""## Resume

This run {resume.get("mode")}.

- Source checkpoint: `{resume.get("from_checkpoint")}`
- Resumed step: {resume.get("from_step")}
- Target total steps: {resume.get("target_total_steps")}
- Additional optimizer steps: {resume.get("additional_steps")}
- Resumed validation loss: {resume.get("resumed_validation_loss")}

{resume.get("note", "")}

The logged validation loss is the {result["eval_batches"]}-batch estimate used to pick the best checkpoint. The story scoreboard measures the full validation split separately.
"""
    else:
        resume_section = ""
    first = history[0]
    last = history[-1]
    step_1000 = next((item for item in history if item["step"] == 1000), None)
    if step_1000 is None:
        curve_reading = "The history does not contain a step-1000 validation point."
    else:
        gained = step_1000["validation_loss"] - result["best_validation_loss"]
        curve_reading = (
            f"Validation loss at step 1000 is {step_1000['validation_loss']:.4f}. "
            f"The best logged validation loss is {result['best_validation_loss']:.4f} "
            f"at step {result['best_step']}. "
            f"The final logged validation loss is {last['validation_loss']:.4f} at step {last['step']}. "
            f"The best point is {gained:.4f} lower than step 1000."
        )
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
- Tokenizer type: {result.get("tokenizer_type", "char")}
- Position encoding: {result.get("position_encoding", "absolute")}
- Vocabulary size: {result["vocab_size"]}
- Block size: {result["block_size"]} tokenizer units
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
- Targets per batch: {tokens_per_step:,}
- Planned target tokens over the full step budget: {exposure:,}

{resume_section}
## Training Setup

AdamW updates the model on next-token cross-entropy. Each logged train loss is the average of the training steps since the previous log. Validation loss is the average over {result["eval_batches"]} validation batches, with gradients disabled. Checkpoints are `{result.get("latest_checkpoint_path", config.LATEST_CHECKPOINT)}` and `{result.get("best_checkpoint_path", config.BEST_CHECKPOINT)}`. The best checkpoint is the lowest logged validation loss, which can be earlier than the final step.

## Loss Progress

![Loss curve]({curve_name})

| Step | Train Loss | Validation Loss | Perplexity | Mean Grad Norm | Elapsed Time |
|---:|---:|---:|---:|---:|---:|
{rows}

## Validation Curve

{curve_reading}

## Best Validation Result

- Step: {result["best_step"]}
- Validation loss: {result["best_validation_loss"]:.4f}
- Validation perplexity: {result["best_validation_perplexity"]:.2f}

Perplexity is `exp(validation loss)`. A lower value means the model assigns more probability to the actual next tokens. It is not a complete measure of story quality.

## Computational Performance

- Training throughput: {result["tokens_per_second"]:,.0f} tokens/sec
- Characters/sec: {"N/A" if result.get("characters_per_second") is None else f"{result['characters_per_second']:,.0f}"}
- Peak GPU memory: {memory}
- Elapsed time, including validation: {result["elapsed_seconds"]:.1f} s

Throughput counts only optimizer steps. Validation time is excluded. Tokens/sec is not directly comparable across character and BPE tokenizers; prefer characters/sec for text-throughput comparisons.

## Observations

{observation} Training loss moved from {first["train_loss"]:.4f} to {last["train_loss"]:.4f}. These curves show that the training pipeline runs. They do not show that the model can write stories yet.

## Limitations

- The run uses a 5,000-story TinyStories slice.
- Perplexity summarizes next-token loss for the active tokenizer units.
- A short run can still leave the validation loss far from a useful language model.
- Block size {result["block_size"]} means different amounts of text for character vs BPE tokens.

## Next Step

Generate text from `{result.get("best_checkpoint_path", config.BEST_CHECKPOINT)}` with the matching saved tokenizer. Do not load character-model weights into a BPE vocabulary.
"""


def save_training_artifacts(result: dict) -> None:
    history_path = Path(config.HISTORY_PATH)
    report_path = Path(config.REPORT_PATH)
    curve_path = Path(config.LOSS_CURVE_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    save_loss_curve(result["history"], curve_path)
    history_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(generate_training_report(result), encoding="utf-8")
