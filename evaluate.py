"""Final evaluation for MiniGPT v1. Does not retrain the model."""

import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from generate import generate_once, load_model
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import Tokenizer
from src.trainer import language_model_loss, perplexity

EVAL_PROMPTS = [
    "Once there was",
    "The little girl",
    "One day a boy",
    "The dog was",
]

BENCHMARK_FILES = {
    "single_head": "experiments/single_head_benchmark.json",
    "single_head_128": "experiments/single_head_128_benchmark.json",
    "multi_head": "experiments/multi_head_benchmark.json",
    "transformer_block": "experiments/transformer_block_benchmark.json",
    "minigpt_forward": "experiments/minigpt_forward_benchmark.json",
}


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_json(path: str | Path) -> dict | list | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def average_validation_loss(
    model: MiniGPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int]:
    """Average next-token cross-entropy over validation batches."""
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            loss = language_model_loss(model(inputs.to(device)), targets.to(device))
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError(f"Validation loss became non-finite: {value}")
            total += value
            count += 1
    if count == 0:
        raise RuntimeError("No validation batches were evaluated.")
    return total / count, count


def generation_diagnostics(text: str, prompt: str) -> dict:
    """Simple character-level diagnostics for generated text."""
    generated = text[len(prompt) :] if text.startswith(prompt) else text
    unique_chars = len(set(generated))
    total_chars = len(generated)
    unique_ratio = unique_chars / total_chars if total_chars else 0.0

    max_run = 1
    current_run = 1
    for index in range(1, len(generated)):
        if generated[index] == generated[index - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    # Count the strongest short-substring repeat of length 3.
    substring_hits = 0
    for size in (3, 4):
        if len(generated) < size * 2:
            continue
        counts: dict[str, int] = {}
        for start in range(0, len(generated) - size + 1):
            piece = generated[start : start + size]
            counts[piece] = counts.get(piece, 0) + 1
        if counts:
            substring_hits = max(substring_hits, max(counts.values()))

    return {
        "total_characters": total_chars,
        "unique_character_count": unique_chars,
        "unique_character_ratio": unique_ratio,
        "max_repeated_character_run": max_run,
        "max_short_substring_repeats": substring_hits,
        "excessive_repetition": max_run >= 8 or substring_hits >= 12,
    }


def vocabulary_valid(model: MiniGPT, tokenizer: Tokenizer, text: str) -> bool:
    ids = tokenizer.encode(text)
    return all(0 <= token_id < model.vocab_size for token_id in ids)


def summarize_training(history_payload: dict | None) -> dict:
    if history_payload is None or "history" not in history_payload:
        return {"available": False}
    history = history_payload["history"]
    if not history:
        return {"available": False}
    first = history[0]
    last = history[-1]
    best_step = history_payload.get("best_step")
    best_validation = history_payload.get("best_validation_loss")
    if best_validation is None:
        best_entry = min(history, key=lambda item: item["validation_loss"])
        best_step = best_entry["step"]
        best_validation = best_entry["validation_loss"]

    absolute_improvement = first["validation_loss"] - best_validation
    percentage = (
        100.0 * absolute_improvement / first["validation_loss"]
        if first["validation_loss"]
        else 0.0
    )
    gap = last["train_loss"] - last["validation_loss"]

    # Flag only when validation clearly worsens while training keeps improving.
    possible_overfitting = False
    if len(history) >= 3:
        best_index = min(range(len(history)), key=lambda index: history[index]["validation_loss"])
        if best_index < len(history) - 1:
            later = history[best_index + 1 :]
            train_improves = last["train_loss"] < history[best_index]["train_loss"] - 0.05
            valid_worsens = last["validation_loss"] > best_validation + 0.1
            possible_overfitting = train_improves and valid_worsens and len(later) >= 2

    seen = set()
    compact = []
    for item in history:
        if item["step"] in (0, last["step"]) or item["step"] % 200 == 0:
            if item["step"] not in seen:
                compact.append(item)
                seen.add(item["step"])

    return {
        "available": True,
        "initial_train_loss": first["train_loss"],
        "initial_validation_loss": first["validation_loss"],
        "final_train_loss": last["train_loss"],
        "final_validation_loss": last["validation_loss"],
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "total_training_time_seconds": history_payload.get("elapsed_seconds", last["elapsed_seconds"]),
        "absolute_validation_loss_improvement": absolute_improvement,
        "percentage_validation_loss_improvement": percentage,
        "train_validation_gap": gap,
        "possible_overfitting": possible_overfitting,
        "selected_steps": compact,
        "tokens_per_second": history_payload.get("tokens_per_second"),
        "peak_gpu_memory_mb": history_payload.get("peak_gpu_memory_mb"),
        "train_stories": history_payload.get("train_stories"),
        "valid_stories": history_payload.get("valid_stories"),
        "train_tokens": history_payload.get("train_tokens"),
        "valid_tokens": history_payload.get("valid_tokens"),
        "batch_size": history_payload.get("batch_size"),
        "learning_rate": history_payload.get("learning_rate"),
        "max_steps": history_payload.get("max_steps"),
        "parameter_count": history_payload.get("parameter_count"),
    }


def load_benchmarks() -> dict:
    results = {}
    for name, path in BENCHMARK_FILES.items():
        payload = load_json(path)
        if payload is None:
            results[name] = {"available": False, "path": path}
            continue
        results[name] = {
            "available": True,
            "path": path,
            "parameter_count": payload.get("parameter_count"),
            "forward_latency_ms": payload.get("forward_latency_ms"),
            "forward_backward_latency_ms": payload.get("forward_backward_latency_ms"),
            "tokens_per_second": payload.get("tokens_per_second"),
            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
            "model": payload.get("model"),
            "head_dim": payload.get("head_dim"),
            "num_heads": payload.get("num_heads"),
        }
    return results


def fmt(value, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "N/A"
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def build_markdown(result: dict) -> str:
    training = result["training"]
    validation = result["validation"]
    baselines = result["baselines"]
    generation = result["generation"]
    sampling = result["sampling_experiments"]
    attention = result["attention_comparison"]
    performance = result["performance"]
    model = result["model"]
    dataset = result["dataset"]

    loss_curve = "![Loss curve](loss_curve.png)" if Path(config.LOSS_CURVE_PATH).exists() else "_Loss curve unavailable._"

    train_rows = ""
    if training.get("available"):
        train_rows = "\n".join(
            f"| {item['step']} | {item['train_loss']:.4f} | {item['validation_loss']:.4f} |"
            for item in training["selected_steps"]
        )

    sample_sections = []
    for item in generation["prompt_samples"]:
        sample_sections.append(
            f"### Prompt: {item['prompt']}\n\n"
            f"**Generated text**\n\n"
            f"```text\n{item['generated_text'][:800]}\n```\n"
        )

    sampling_rows = "\n".join(
        f"| {item['temperature']} | "
        f"{'None' if item['top_k'] is None else item['top_k']} | "
        f"{item['tokens_per_second']:.1f} | "
        f"{item['diagnostics']['unique_character_ratio']:.3f} |"
        for item in sampling["temperature_and_topk"]
    )

    attention_rows = "\n".join(
        f"| {label} | {fmt(row.get('parameter_count'), 0)} | "
        f"{fmt(row.get('forward_latency_ms'), 3, ' ms')} | "
        f"{fmt(row.get('forward_backward_latency_ms'), 3, ' ms')} | "
        f"{fmt(row.get('tokens_per_second'), 0, ' tok/s')} | "
        f"{fmt(row.get('peak_gpu_memory_mb'), 2, ' MB')} |"
        for label, row in attention["rows"].items()
    )

    overfitting_note = (
        "Validation loss did not show a clear overfitting pattern in this short run."
        if not training.get("possible_overfitting")
        else "Validation loss worsened after the best step while training loss kept improving."
    )

    return f"""# MiniGPT From Scratch — Final Report

## 1. Project Overview

MiniGPT is a character-level GPT-style language model trained on a TinyStories subset. This report summarizes architecture, training, validation, generation, and compute measurements for version 1. Detailed intermediate benchmarks live under `experiments/`.

## 2. Architecture

```text
Text
↓
Character Tokenizer
↓
Token IDs
↓
Token Embedding + Position Embedding
↓
Transformer Block × {model["num_layers"]}
│
├── LayerNorm
├── Multi-Head Causal Self-Attention
├── Residual
├── LayerNorm
├── Feed Forward
└── Residual
↓
Final LayerNorm
↓
LM Head
↓
Vocabulary Logits
```

| Setting | Value |
|---|---:|
| Vocabulary size | {model["vocab_size"]} |
| Block size | {model["block_size"]} |
| Embedding dimension | {model["embedding_dim"]} |
| Heads | {model["num_heads"]} |
| Head dimension | {model["head_dim"]} |
| Layers | {model["num_layers"]} |
| Dropout | {model["dropout"]} |
| Parameters | {model["parameter_count"]:,} ({model["parameter_count_millions"]:.2f}M) |

## 3. Dataset

| Item | Value |
|---|---:|
| Training stories | {fmt(dataset["train_stories"], 0)} |
| Validation stories | {fmt(dataset["valid_stories"], 0)} |
| Training tokens | {fmt(dataset["train_tokens"], 0)} |
| Validation tokens | {fmt(dataset["valid_tokens"], 0)} |
| Vocabulary size | {dataset["vocab_size"]} |
| Block size | {dataset["block_size"]} |
| Batch size | {fmt(dataset["batch_size"], 0)} |

## 4. Training Configuration

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | {fmt(training.get("learning_rate"), 6)} |
| Max steps | {fmt(training.get("max_steps"), 0)} |
| Gradient clip | {config.GRAD_CLIP} |
| Device | `{result["device"]}` |
| GPU | {result["gpu_name"] or "none"} |

## 5. Training Results

{loss_curve}

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
{train_rows}

- Best validation loss: {fmt(training.get("best_validation_loss"))} at step {fmt(training.get("best_step"), 0)}
- Absolute validation improvement: {fmt(training.get("absolute_validation_loss_improvement"))}
- Percentage validation improvement: {fmt(training.get("percentage_validation_loss_improvement"), 1, "%")}
- Final train-validation gap: {fmt(training.get("train_validation_gap"))}
- Overfitting note: {overfitting_note}

## 6. Validation and Perplexity

Evaluated over {validation["batches_evaluated"]} validation batches.

| Metric | Trained MiniGPT |
|---|---:|
| Validation loss | {validation["loss"]:.4f} |
| Validation perplexity | {validation["perplexity"]:.2f} |

## 7. Baseline Comparison

| Model | Validation Loss | Perplexity |
|---|---:|---:|
| Uniform Random Baseline | {baselines["uniform_loss"]:.4f} | {baselines["uniform_perplexity"]:.2f} |
| Untrained MiniGPT | {baselines["untrained_loss"]:.4f} | {baselines["untrained_perplexity"]:.2f} |
| Trained MiniGPT | {baselines["trained_loss"]:.4f} | {baselines["trained_perplexity"]:.2f} |

Training improved next-token prediction relative to both the uniform baseline and a randomly initialized copy of the same architecture.

## 8. Single-Head vs Multi-Head Attention

These numbers are module-level compute benchmarks, not trained language quality.

| Variant | Parameters | Forward | Forward + Backward | Throughput | Peak GPU Memory |
|---|---:|---:|---:|---:|---:|
{attention_rows}

Single Head 32 has less attention width than Multi Head 4×32. Single Head 128 matches total attention width 128 and is the fairer architectural comparison. Neither table alone proves which design learns TinyStories better.

## 9. Generation Results

Main configuration: `temperature = {generation["temperature"]}`, `top_k = {generation["top_k"]}`, `max_new_tokens = {generation["max_new_tokens"]}`, seed `{generation["seed"]}`.

{chr(10).join(sample_sections)}

- All generated token IDs were vocabulary-valid: {generation["all_vocabulary_valid"]}
- Diversity from 5 seeds on one prompt: {generation["diversity"]["unique_outputs"]} unique outputs, average unique-character ratio {generation["diversity"]["average_unique_character_ratio"]:.3f}

## 10. Sampling Experiments

Fixed prompt: `{sampling["prompt"]}`

| Temperature | Top-k | Generation Speed | Unique Character Ratio |
|---:|---:|---:|---:|
{sampling_rows}

## 11. Computational Performance

| Metric | Result |
|---|---:|
| Model Parameters | {fmt(performance["parameter_count"], 0)} |
| Training Throughput | {fmt(performance["training_tokens_per_second"], 0, " tokens/sec")} |
| Generation Throughput | {fmt(performance["generation_tokens_per_second"], 1, " tokens/sec")} |
| Peak Training GPU Memory | {fmt(performance["peak_training_gpu_memory_mb"], 2, " MB")} |
| Peak Inference GPU Memory | {fmt(performance["peak_inference_gpu_memory_mb"], 2, " MB")} |

## 12. Limitations

- Character-level tokenizer with a vocabulary of {model["vocab_size"]} characters
- Context window of only {model["block_size"]} tokens
- Small model size ({model["parameter_count_millions"]:.2f}M parameters)
- Limited TinyStories slice and short training duration
- No pretrained weights
- No semantic benchmark beyond next-character loss and sample inspection
- Basic sampling: temperature and top-k only
- No KV cache, mixed precision, or flash attention

## 13. Future Improvements

- BPE tokenizer
- Larger context window
- Larger model
- Longer TinyStories training
- Learning-rate scheduler
- Mixed precision
- KV cache
- Top-p sampling
- Better evaluation metrics
- Larger dataset
"""


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    tokenizer = Tokenizer.load(config.TOKENIZER_PATH)
    model, checkpoint = load_model(config.BEST_CHECKPOINT, device)
    model_config = checkpoint["model_config"]

    # Validation text is encoded with the saved training vocabulary.
    _train_stories, valid_stories = load_tinystories(num_train=1, num_valid=500)
    valid_text = prepare_text(valid_stories)
    valid_ids = tokenizer.encode(valid_text)
    valid_loader = DataLoader(
        TextDataset(valid_ids, model.block_size),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    print("Evaluating trained model on validation data...")
    trained_loss, batches = average_validation_loss(
        model, valid_loader, device, config.EVAL_MAX_BATCHES
    )
    trained_ppl = perplexity(trained_loss)

    print("Evaluating untrained baseline...")
    untrained = MiniGPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)
    untrained_loss, _ = average_validation_loss(
        untrained, valid_loader, device, config.EVAL_MAX_BATCHES
    )
    untrained_ppl = perplexity(untrained_loss)

    vocab_size = model.vocab_size
    uniform_loss = math.log(vocab_size)
    uniform_ppl = float(vocab_size)

    history_payload = load_json(config.HISTORY_PATH)
    training = summarize_training(history_payload)
    benchmarks = load_benchmarks()

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_dim = model.embedding_dim // model.num_heads

    print("Running standard generation prompts...")
    prompt_samples = []
    for prompt in EVAL_PROMPTS:
        sample = generate_once(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            temperature=0.8,
            top_k=20,
            seed=config.GENERATION_SEED,
            device=device,
        )
        diagnostics = generation_diagnostics(sample["generated_text"], prompt)
        sample["diagnostics"] = diagnostics
        sample["vocabulary_valid"] = vocabulary_valid(model, tokenizer, sample["generated_text"])
        prompt_samples.append(sample)
        safe_print(f"[{prompt}] {sample['generated_text'][:160]}")

    print("Running sampling sensitivity experiments...")
    sampling_prompt = "Once there was"
    sampling_runs = []
    for temperature, top_k in (
        (0.5, 20),
        (0.8, 20),
        (1.0, 20),
        (1.2, 20),
        (0.8, 5),
        (0.8, 20),
        (0.8, 40),
        (0.8, None),
    ):
        # Deduplicate the repeated 0.8/20 configuration.
        key = (temperature, top_k)
        if any(item["temperature"] == key[0] and item["top_k"] == key[1] for item in sampling_runs):
            continue
        sample = generate_once(
            model,
            tokenizer,
            prompt=sampling_prompt,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            temperature=temperature,
            top_k=top_k,
            seed=config.GENERATION_SEED,
            device=device,
        )
        sample["diagnostics"] = generation_diagnostics(sample["generated_text"], sampling_prompt)
        sample["vocabulary_valid"] = vocabulary_valid(model, tokenizer, sample["generated_text"])
        sampling_runs.append(sample)

    print("Measuring generation diversity across seeds...")
    diversity_samples = []
    for seed in range(config.GENERATION_SEED, config.GENERATION_SEED + 5):
        sample = generate_once(
            model,
            tokenizer,
            prompt=sampling_prompt,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            temperature=0.8,
            top_k=20,
            seed=seed,
            device=device,
        )
        sample["diagnostics"] = generation_diagnostics(sample["generated_text"], sampling_prompt)
        diversity_samples.append(sample)
    unique_outputs = len({item["generated_text"] for item in diversity_samples})
    average_unique_ratio = sum(
        item["diagnostics"]["unique_character_ratio"] for item in diversity_samples
    ) / len(diversity_samples)

    all_valid = all(item["vocabulary_valid"] for item in prompt_samples + sampling_runs)
    generation_speeds = [item["tokens_per_second"] for item in prompt_samples]
    inference_memory = [
        item["peak_gpu_memory_mb"]
        for item in prompt_samples + sampling_runs
        if item["peak_gpu_memory_mb"] is not None
    ]

    attention_rows = {
        "Single Head 32": benchmarks["single_head"] if benchmarks["single_head"]["available"] else {},
        "Single Head 128": benchmarks["single_head_128"] if benchmarks["single_head_128"]["available"] else {},
        "Multi Head 4×32": benchmarks["multi_head"] if benchmarks["multi_head"]["available"] else {},
    }

    dataset = {
        "train_stories": training.get("train_stories"),
        "valid_stories": training.get("valid_stories") or len(valid_stories),
        "train_tokens": training.get("train_tokens"),
        "valid_tokens": training.get("valid_tokens") or len(valid_ids),
        "vocab_size": vocab_size,
        "block_size": model.block_size,
        "batch_size": training.get("batch_size") or config.BATCH_SIZE,
    }

    result = {
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": {
            "path": config.BEST_CHECKPOINT,
            "step": checkpoint.get("step"),
            "train_loss": checkpoint.get("train_loss"),
            "validation_loss": checkpoint.get("validation_loss"),
        },
        "model": {
            "vocab_size": vocab_size,
            "block_size": model.block_size,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "head_dim": head_dim,
            "num_layers": model.num_layers,
            "dropout": model_config["dropout"],
            "parameter_count": parameter_count,
            "parameter_count_millions": parameter_count / 1_000_000,
        },
        "dataset": dataset,
        "training": training,
        "validation": {
            "loss": trained_loss,
            "perplexity": trained_ppl,
            "batches_evaluated": batches,
        },
        "baselines": {
            "uniform_loss": uniform_loss,
            "uniform_perplexity": uniform_ppl,
            "untrained_loss": untrained_loss,
            "untrained_perplexity": untrained_ppl,
            "trained_loss": trained_loss,
            "trained_perplexity": trained_ppl,
        },
        "generation": {
            "temperature": 0.8,
            "top_k": 20,
            "max_new_tokens": config.GENERATION_MAX_NEW_TOKENS,
            "seed": config.GENERATION_SEED,
            "prompt_samples": prompt_samples,
            "all_vocabulary_valid": all_valid,
            "diversity": {
                "samples": len(diversity_samples),
                "unique_outputs": unique_outputs,
                "average_unique_character_ratio": average_unique_ratio,
            },
        },
        "sampling_experiments": {
            "prompt": sampling_prompt,
            "temperature_and_topk": sampling_runs,
        },
        "performance": {
            "parameter_count": parameter_count,
            "training_tokens_per_second": training.get("tokens_per_second"),
            "generation_tokens_per_second": (
                sum(generation_speeds) / len(generation_speeds) if generation_speeds else None
            ),
            "peak_training_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
            "peak_inference_gpu_memory_mb": max(inference_memory) if inference_memory else None,
        },
        "attention_comparison": {
            "rows": attention_rows,
            "benchmarks": benchmarks,
        },
    }

    json_path = Path(config.FINAL_EVAL_JSON)
    report_path = Path(config.FINAL_EVAL_REPORT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_markdown(result), encoding="utf-8")

    print()
    print("Validation loss:", f"{trained_loss:.4f}")
    print("Validation perplexity:", f"{trained_ppl:.2f}")
    print("Untrained loss:", f"{untrained_loss:.4f}")
    print("Uniform baseline loss:", f"{uniform_loss:.4f}")
    print("Saved final evaluation JSON:")
    print(json_path.as_posix())
    print("Saved final report:")
    print(report_path.as_posix())


if __name__ == "__main__":
    main()
