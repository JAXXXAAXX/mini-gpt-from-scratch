"""Computational benchmark for an attention module.

The timer is not tied to one class. A later multi-head module can be passed
to ``measure_module`` with the same batch size, sequence length, and
embedding dimension. Learning quality is a separate record: this file only
reserves a slot for it until a language-model head and training loop exist.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

# Fixed inputs for every single-head vs multi-head comparison.
COMPARISON_CONFIG = {
    "batch_size": 32,
    "sequence_length": 64,
    "embedding_dim": 128,
    "head_dim": 32,
    "dropout": 0.1,
    "warmup_iterations": 20,
    "measured_iterations": 100,
    "learning_steps": 1000,
}

JSON_PATH = Path("experiments/single_head_benchmark.json")
MARKDOWN_PATH = Path("experiments/single_head_benchmark.md")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_module(
    module: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    warmup_iterations: int,
    measured_iterations: int,
    device: torch.device,
) -> dict:
    """Time forward, then forward plus backward, for any [B, T, C] module."""
    module = module.to(device)
    parameter_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    tokens = batch_size * sequence_length
    x = torch.randn(batch_size, sequence_length, embedding_dim, device=device)

    module.eval()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            module(x)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(measured_iterations):
            module(x)
        _synchronize(device)
        forward_seconds = (time.perf_counter() - start) / measured_iterations

    module.train()
    for _ in range(warmup_iterations):
        module.zero_grad(set_to_none=True)
        module(x).mean().backward()
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(measured_iterations):
        module.zero_grad(set_to_none=True)
        module(x).mean().backward()
    _synchronize(device)
    step_seconds = (time.perf_counter() - start) / measured_iterations

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mb = None

    return {
        "parameter_count": parameter_count,
        "forward_latency_ms": forward_seconds * 1000,
        "forward_backward_latency_ms": step_seconds * 1000,
        # One timed training step processes one batch of tokens.
        "tokens_per_second": tokens / step_seconds,
        "peak_gpu_memory_mb": peak_mb,
        "input_shape": [batch_size, sequence_length, embedding_dim],
    }


def verify_causal_mask(module: nn.Module, embedding_dim: int, device: torch.device) -> dict:
    """Check a length-4 sequence. Future attention probabilities must be ~0."""
    was_training = module.training
    module.eval()
    sequence_length = 4
    probe = torch.randn(1, sequence_length, embedding_dim, device=device)
    with torch.no_grad():
        _output, details = module(probe, return_details=True)
    weights = details["attention_weights"][0].detach().float().cpu()
    module.train(was_training)

    future = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
    future_near_zero = bool(torch.all(weights[future] < 1e-5))
    position_checks = []
    for index in range(sequence_length):
        allowed_mass = float(weights[index, : index + 1].sum())
        blocked_mass = float(weights[index, index + 1 :].sum()) if index + 1 < sequence_length else 0.0
        position_checks.append(
            {
                "position": index,
                "allowed_positions": list(range(index + 1)),
                "allowed_mass": allowed_mass,
                "blocked_mass": blocked_mass,
                "passed": abs(allowed_mass - 1.0) < 1e-4 and blocked_mass < 1e-5,
            }
        )

    return {
        "passed": future_near_zero and all(item["passed"] for item in position_checks),
        "sequence_length": sequence_length,
        "future_probabilities_near_zero": future_near_zero,
        "position_checks": position_checks,
        "weight_matrix": [[round(float(value), 4) for value in row] for row in weights],
    }


def learning_comparison_placeholder() -> dict:
    """Reserved comparison. Do not fill losses until training exists.

    TODO: train single-head and multi-head models for the same number of
    steps, on the same TinyStories slice, with the same batch size, block
    size, embedding dimension, optimizer, and learning rate. Record train
    loss, validation loss, and elapsed training time separately from the
    speed and memory numbers above.
    """
    pending = {
        "train_loss": None,
        "validation_loss": None,
        "elapsed_training_time_sec": None,
        "loss_after_fixed_steps": None,
    }
    return {
        "status": "not_available",
        "fixed_training_steps": COMPARISON_CONFIG["learning_steps"],
        "shared_settings": {
            "dataset": "TinyStories subset",
            "batch_size": COMPARISON_CONFIG["batch_size"],
            "block_size": COMPARISON_CONFIG["sequence_length"],
            "embedding_dim": COMPARISON_CONFIG["embedding_dim"],
            "optimizer": "not implemented yet",
            "learning_rate": None,
            "training_steps": COMPARISON_CONFIG["learning_steps"],
        },
        "single_head": dict(pending),
        "multi_head": dict(pending),
    }


def device_metadata(device: torch.device) -> dict:
    cuda = device.type == "cuda"
    return {
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda else None,
    }


def _shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def shapes_from_details(token_ids: torch.Tensor, embedded: torch.Tensor, details: dict) -> dict:
    return {
        "token_ids": _shape(token_ids),
        "embedded_input": _shape(embedded),
        "q": _shape(details["q"]),
        "k": _shape(details["k"]),
        "v": _shape(details["v"]),
        "attention_scores": _shape(details["attention_scores"]),
        "attention_weights": _shape(details["attention_weights"]),
        "output": _shape(details["output"]),
    }


def assemble_result(measured: dict, shapes: dict, causal_mask: dict, meta: dict) -> dict:
    """One dictionary feeds both the JSON file and the Markdown report."""
    config = COMPARISON_CONFIG
    return {
        "model": "single_head_attention",
        "batch_size": config["batch_size"],
        "sequence_length": config["sequence_length"],
        "embedding_dim": config["embedding_dim"],
        "head_dim": config["head_dim"],
        "dropout": config["dropout"],
        "warmup_iterations": config["warmup_iterations"],
        "measured_iterations": config["measured_iterations"],
        **meta,
        **measured,
        "shapes": shapes,
        "causal_mask": causal_mask,
        "learning_comparison": learning_comparison_placeholder(),
    }


def _fmt_number(value, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _memory_text(result: dict) -> str:
    if result["peak_gpu_memory_mb"] is None:
        return "GPU memory measurement is unavailable because CUDA was not used."
    return (
        "Peak allocated GPU memory during the timed forward + backward steps was "
        f"{result['peak_gpu_memory_mb']:.2f} MB."
    )


def generate_benchmark_report(result: dict) -> str:
    """Render a Markdown report from a benchmark result dictionary."""
    shapes = result["shapes"]
    mask = result["causal_mask"]
    matrix = mask["weight_matrix"]
    matrix_rows = "\n".join(
        "| "
        + str(index)
        + " | "
        + " | ".join(f"{value:.4f}" for value in row)
        + " |"
        for index, row in enumerate(matrix)
    )
    position_lines = "\n".join(
        f"- Position {item['position']} can attend to {item['allowed_positions']}: "
        f"{'yes' if item['passed'] else 'no'}"
        for item in mask["position_checks"]
    )
    gpu_name = result["gpu_name"] or "none"
    cuda_version = result["cuda_version"] or "none"
    memory_cell = (
        "unavailable" if result["peak_gpu_memory_mb"] is None else f"{result['peak_gpu_memory_mb']:.2f} MB"
    )
    single_memory = "TBD" if result["peak_gpu_memory_mb"] is None else f"{result['peak_gpu_memory_mb']:.2f} MB"

    return f"""# Single-Head Self-Attention Benchmark Report

## Experiment Configuration

- Device: `{result["device"]}`
- GPU name: {gpu_name}
- PyTorch version: {result["pytorch_version"]}
- CUDA version: {cuda_version}
- Batch size: {result["batch_size"]}
- Sequence length: {result["sequence_length"]}
- Embedding dimension: {result["embedding_dim"]}
- Head dimension: {result["head_dim"]}
- Dropout: {result["dropout"]}
- Warm-up iterations: {result["warmup_iterations"]}
- Measured iterations: {result["measured_iterations"]}
- Forward timing runs in eval mode. Forward + backward timing runs in train mode.
- Throughput is `(batch_size × sequence_length) / forward_backward_seconds`.

## Model Configuration

- Model name: `{result["model"]}`
- Trainable parameters: {result["parameter_count"]:,}
- Input token IDs: `{shapes["token_ids"]}`
- Embedded input X: `{shapes["embedded_input"]}`
- Q: `{shapes["q"]}`
- K: `{shapes["k"]}`
- V: `{shapes["v"]}`
- Attention scores: `{shapes["attention_scores"]}`
- Attention weights: `{shapes["attention_weights"]}`
- Output: `{shapes["output"]}`

## Computational Performance

| Metric | Result |
|---|---:|
| Trainable Parameters | {result["parameter_count"]:,} |
| Forward Latency | {result["forward_latency_ms"]:.3f} ms |
| Forward + Backward Latency | {result["forward_backward_latency_ms"]:.3f} ms |
| Throughput | {_fmt_number(result["tokens_per_second"], 0)} tokens/sec |
| Peak GPU Memory | {memory_cell} |

## Causal Mask Verification

Status: {"passed" if mask["passed"] else "failed"}.

Future attention probabilities are {"approximately zero" if mask["future_probabilities_near_zero"] else "not approximately zero"}.

{position_lines}

Example attention weights for one sequence of length {mask["sequence_length"]}. Rows are queries and columns are keys. Entries above the diagonal are the masked future positions.

| | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
{matrix_rows}

## Shape Verification

| Tensor | Shape | Measured |
|---|---|---|
| Input token IDs | `[B, T]` | `{shapes["token_ids"]}` |
| Embedded input X | `[B, T, C]` | `{shapes["embedded_input"]}` |
| Q | `[B, T, H]` | `{shapes["q"]}` |
| K | `[B, T, H]` | `{shapes["k"]}` |
| V | `[B, T, H]` | `{shapes["v"]}` |
| Attention Scores | `[B, T, T]` | `{shapes["attention_scores"]}` |
| Attention Weights | `[B, T, T]` | `{shapes["attention_weights"]}` |
| Attention Output | `[B, T, H]` | `{shapes["output"]}` |

## Interpretation

One causal self-attention head has {result["parameter_count"]:,} trainable parameters. On this run the forward + backward step processed about {_fmt_number(result["tokens_per_second"], 0)} tokens/sec, using a batch of {result["batch_size"]} sequences of length {result["sequence_length"]}. {_memory_text(result)}

This is a computational baseline for one attention module. It is not evidence that single-head attention learns better or worse than multi-head attention.

## Limitations

This benchmark measures only the single-head attention module. It does not measure full language-model quality.

- There is no complete Transformer block yet.
- There is no language-model head yet.
- Validation loss comparison is not yet available.
- Runtime measurements can vary with GPU state and other system load.

## Future Comparison with Multi-Head Attention

Computational performance and learning performance stay in separate columns. They are not combined into one score.

| Metric | Single Head | Multi Head |
|---|---:|---:|
| Parameters | {result["parameter_count"]:,} | TBD |
| Forward Latency | {result["forward_latency_ms"]:.3f} ms | TBD |
| Forward + Backward Latency | {result["forward_backward_latency_ms"]:.3f} ms | TBD |
| Throughput | {_fmt_number(result["tokens_per_second"], 0)} tokens/sec | TBD |
| Peak GPU Memory | {single_memory} | TBD |
| Validation Loss | TBD | TBD |
"""


def save_benchmark(result: dict, json_path: Path = JSON_PATH, markdown_path: Path = MARKDOWN_PATH) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(generate_benchmark_report(result), encoding="utf-8")


def print_baseline_summary(result: dict) -> None:
    memory = (
        "unavailable"
        if result["peak_gpu_memory_mb"] is None
        else f"{result['peak_gpu_memory_mb']:.2f} MB"
    )
    print("=== Single-Head Attention Baseline ===")
    print(f"Parameters:               {result['parameter_count']:,}")
    print(f"Forward latency:          {result['forward_latency_ms']:.3f} ms")
    print(f"Forward + backward:       {result['forward_backward_latency_ms']:.3f} ms")
    print(f"Throughput:               {result['tokens_per_second']:,.0f} tokens/sec")
    print(f"Peak GPU memory:          {memory}")
