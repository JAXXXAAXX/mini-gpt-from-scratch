"""Load a trained MiniGPT checkpoint and sample text from a prompt."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import config
from src.model import MiniGPT
from src.tokenizer import Tokenizer


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT.")
    parser.add_argument("--prompt", default=config.GENERATION_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=config.GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.GENERATION_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.GENERATION_TOP_K)
    parser.add_argument("--checkpoint", default=config.BEST_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run the fixed temperature and top-k comparison suite.",
    )
    return parser.parse_args()


def set_seed(seed: int | None, device: torch.device) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MiniGPT, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = MiniGPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def encode_prompt(tokenizer: Tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    # Reuse the training vocabulary. Do not fit a new tokenizer for generation.
    try:
        token_ids = tokenizer.encode(prompt)
    except ValueError as exc:
        raise ValueError(
            "The prompt contains characters that are not in the training vocabulary.\n"
            f"{exc}"
        ) from exc
    if not token_ids:
        raise ValueError("The prompt encoded to an empty token sequence.")
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def generate_once(
    model: MiniGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
    device: torch.device,
) -> dict:
    set_seed(seed, device)
    prompt_ids = encode_prompt(tokenizer, prompt, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if output_ids.shape[1] != prompt_ids.shape[1] + max_new_tokens:
        raise RuntimeError(
            f"Expected {prompt_ids.shape[1] + max_new_tokens} tokens, got {output_ids.shape[1]}."
        )
    if model.training:
        raise RuntimeError("The model should remain in evaluation mode after generation.")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Generation stored gradients. Use inference mode with no backward pass.")

    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids)
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "generated_text": text,
        "generation_time_seconds": elapsed,
        "tokens_generated": max_new_tokens,
        "tokens_per_second": max_new_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_gpu_memory_mb": peak_mb,
    }


def _report_text(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated in report; full text is in generation_samples.json]"


def generate_report(samples: list[dict], checkpoint: dict, model: MiniGPT) -> str:
    rows = "\n".join(
        f"| {chr(ord('A') + index)} | {item['temperature']} | "
        f"{'None' if item['top_k'] is None else item['top_k']} | "
        f"{item['tokens_generated']} | {item['generation_time_seconds']:.3f} s | "
        f"{item['tokens_per_second']:.1f} |"
        for index, item in enumerate(samples)
    )
    sample_sections = []
    for index, item in enumerate(samples):
        label = chr(ord("A") + index)
        sample_sections.append(
            f"### Experiment {label}\n\n"
            f"**Prompt**\n\n"
            f"> {item['prompt']}\n\n"
            f"**Generated text**\n\n"
            f"```text\n{_report_text(item['generated_text'])}\n```"
        )
    peak_values = [item["peak_gpu_memory_mb"] for item in samples if item["peak_gpu_memory_mb"] is not None]
    peak_text = (
        f"{max(peak_values):.2f} MB"
        if peak_values
        else "unavailable because CUDA was not used"
    )
    avg_tps = sum(item["tokens_per_second"] for item in samples) / len(samples)
    return f"""# MiniGPT Generation Report

## Model

```text
{model.architecture_summary()}
```

## Checkpoint

- Path: `{config.BEST_CHECKPOINT}`
- Training step: {checkpoint["step"]}
- Train loss: {checkpoint["train_loss"]:.4f}
- Validation loss: {checkpoint["validation_loss"]:.4f}

## Generation Configuration

- Prompt: `{samples[0]["prompt"]}`
- Max new tokens: {samples[0]["max_new_tokens"]}
- Seed: {samples[0]["seed"]}
- Tokenizer: `{config.TOKENIZER_PATH}`

## Sampling Experiments

| Experiment | Temperature | Top-k | Tokens | Time | Tokens/sec |
|---|---:|---:|---:|---:|---:|
{rows}

## Generated Samples

{chr(10).join(sample_sections)}

## Generation Performance

- Average tokens/sec across experiments: {avg_tps:.1f}
- Peak GPU memory during generation: {peak_text}

## Observations

Lower temperature and a smaller top-k keep the next character closer to the model's highest scores. Higher temperature and a larger top-k, or no top-k, let lower-scoring characters appear more often. These samples show sampling behavior after 1,000 training steps. They are not a finished story writer.

## Limitations

- The model uses a character vocabulary and a short context of {model.block_size} tokens.
- There is no top-p sampling, beam search, repetition penalty, or KV cache.
- Generation quality is limited by the short TinyStories training run.
- Unknown prompt characters are rejected instead of being mapped to a fallback token.
"""


def run_experiments(model: MiniGPT, tokenizer: Tokenizer, checkpoint: dict, device: torch.device) -> None:
    experiments = [
        {"temperature": 0.5, "top_k": 10},
        {"temperature": 0.8, "top_k": 20},
        {"temperature": 1.0, "top_k": None},
        {"temperature": 1.2, "top_k": 40},
    ]
    samples = []
    for settings in experiments:
        result = generate_once(
            model,
            tokenizer,
            prompt=config.GENERATION_PROMPT,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            seed=config.GENERATION_SEED,
            device=device,
        )
        samples.append(result)
        print(
            f"temperature={settings['temperature']} top_k={settings['top_k']} "
            f"time={result['generation_time_seconds']:.3f}s "
            f"tokens/sec={result['tokens_per_second']:.1f}"
        )
        safe_print(result["generated_text"][:400])
        print("---")

    samples_path = Path(config.GENERATION_SAMPLES_PATH)
    report_path = Path(config.GENERATION_REPORT_PATH)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(generate_report(samples, checkpoint, model), encoding="utf-8")
    print("Saved generation samples:")
    print(samples_path.as_posix())
    print("Saved generation report:")
    print(report_path.as_posix())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    tokenizer = Tokenizer.load(config.TOKENIZER_PATH)
    model, checkpoint = load_model(args.checkpoint, device)

    if args.run_experiments:
        run_experiments(model, tokenizer, checkpoint, device)
        return

    top_k = None if args.top_k is not None and args.top_k <= 0 else args.top_k
    result = generate_once(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
        device=device,
    )
    print()
    safe_print(result["generated_text"])
    print()
    print(f"tokens/sec: {result['tokens_per_second']:.1f}")
    if result["peak_gpu_memory_mb"] is not None:
        print(f"peak GPU memory: {result['peak_gpu_memory_mb']:.2f} MB")


if __name__ == "__main__":
    main()
