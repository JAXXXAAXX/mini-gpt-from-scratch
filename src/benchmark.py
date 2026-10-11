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
    "num_heads": 4,
    "dropout": 0.1,
    "warmup_iterations": 20,
    "measured_iterations": 100,
    "learning_steps": 1000,
}

JSON_PATH = Path("experiments/single_head_benchmark.json")
MARKDOWN_PATH = Path("experiments/single_head_benchmark.md")
MULTI_JSON_PATH = Path("experiments/multi_head_benchmark.json")
MULTI_MARKDOWN_PATH = Path("experiments/multi_head_benchmark.md")
WIDE_JSON_PATH = Path("experiments/single_head_128_benchmark.json")
WIDE_MARKDOWN_PATH = Path("experiments/single_head_128_benchmark.md")
COMPARISON_MARKDOWN_PATH = Path("experiments/attention_comparison.md")
BLOCK_JSON_PATH = Path("experiments/transformer_block_benchmark.json")
BLOCK_MARKDOWN_PATH = Path("experiments/transformer_block_benchmark.md")
MINIGPT_JSON_PATH = Path("experiments/minigpt_forward_benchmark.json")
MINIGPT_MARKDOWN_PATH = Path("experiments/minigpt_forward_benchmark.md")


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
    vocab_size: int | None = None,
) -> dict:
    """Time forward, then forward plus backward.

    By default the input is a float tensor [B, T, C]. Pass vocab_size when the
    module consumes token ids [B, T], as MiniGPT does.
    """
    module = module.to(device)
    parameter_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    tokens = batch_size * sequence_length
    if vocab_size is None:
        x = torch.randn(batch_size, sequence_length, embedding_dim, device=device)
    else:
        x = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

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


def _check_weight_matrix(weights: torch.Tensor) -> dict:
    """Check one [T, T] attention matrix. Future positions must be ~0."""
    sequence_length = weights.shape[0]
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


def verify_causal_mask(module: nn.Module, embedding_dim: int, device: torch.device) -> dict:
    """Check a length-4 sequence. Future attention probabilities must be ~0.

    A multi-head module returns one weight matrix per head. Every head is checked.
    The example matrix kept for the report is the first head only.
    """
    was_training = module.training
    module.eval()
    sequence_length = 4
    probe = torch.randn(1, sequence_length, embedding_dim, device=device)
    with torch.no_grad():
        _output, details = module(probe, return_details=True)
    raw_weights = details["attention_weights"]
    groups = raw_weights if isinstance(raw_weights, list) else [raw_weights]
    reports = [_check_weight_matrix(group[0].detach().float().cpu()) for group in groups]
    module.train(was_training)

    first = reports[0]
    return {
        "passed": all(item["passed"] for item in reports),
        "head_count": len(reports),
        "heads_passed": [item["passed"] for item in reports],
        "sequence_length": sequence_length,
        "future_probabilities_near_zero": all(
            item["future_probabilities_near_zero"] for item in reports
        ),
        "position_checks": first["position_checks"],
        "weight_matrix": first["weight_matrix"],
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


def assemble_result(
    measured: dict,
    shapes: dict,
    causal_mask: dict,
    meta: dict,
    model: str = "single_head_attention",
    head_dim: int | None = None,
    num_heads: int = 1,
    report_title: str = "Single-Head Self-Attention Benchmark Report",
) -> dict:
    """One dictionary feeds both the JSON file and the Markdown report."""
    config = COMPARISON_CONFIG
    return {
        "model": model,
        "report_title": report_title,
        "batch_size": config["batch_size"],
        "sequence_length": config["sequence_length"],
        "embedding_dim": config["embedding_dim"],
        "head_dim": config["head_dim"] if head_dim is None else head_dim,
        "num_heads": num_heads,
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


def _model_config_lines(result: dict) -> str:
    shapes = result["shapes"]
    lines = [
        f"- Model name: `{result['model']}`",
        f"- Heads: {result.get('num_heads', 1)}",
        f"- Head dimension: {result['head_dim']}",
        f"- Trainable parameters: {result['parameter_count']:,}",
    ]
    if "head_outputs" in shapes:
        lines.append(f"- Input X: `{shapes['input']}`")
        for index, shape in enumerate(shapes["head_outputs"], start=1):
            lines.append(f"- Head {index} output: `{shape}`")
        lines.append(f"- Concatenated output: `{shapes['concatenated']}`")
        lines.append(f"- Projected output: `{shapes['output']}`")
        return "\n".join(lines)
    lines.extend(
        [
            f"- Input token IDs: `{shapes['token_ids']}`",
            f"- Embedded input X: `{shapes['embedded_input']}`",
            f"- Q: `{shapes['q']}`",
            f"- K: `{shapes['k']}`",
            f"- V: `{shapes['v']}`",
            f"- Attention scores: `{shapes['attention_scores']}`",
            f"- Attention weights: `{shapes['attention_weights']}`",
            f"- Output: `{shapes['output']}`",
        ]
    )
    return "\n".join(lines)


def _shape_table(shapes: dict) -> str:
    if "head_outputs" in shapes:
        rows = [f"| Input X | `[B, T, C]` | `{shapes['input']}` |"]
        for index, shape in enumerate(shapes["head_outputs"], start=1):
            rows.append(f"| Head {index} output | `[B, T, H]` | `{shape}` |")
        rows.append(f"| Concatenated output | `[B, T, C]` | `{shapes['concatenated']}` |")
        rows.append(f"| Projected output | `[B, T, C]` | `{shapes['output']}` |")
        body = "\n".join(rows)
    else:
        body = "\n".join(
            [
                f"| Input token IDs | `[B, T]` | `{shapes['token_ids']}` |",
                f"| Embedded input X | `[B, T, C]` | `{shapes['embedded_input']}` |",
                f"| Q | `[B, T, H]` | `{shapes['q']}` |",
                f"| K | `[B, T, H]` | `{shapes['k']}` |",
                f"| V | `[B, T, H]` | `{shapes['v']}` |",
                f"| Attention Scores | `[B, T, T]` | `{shapes['attention_scores']}` |",
                f"| Attention Weights | `[B, T, T]` | `{shapes['attention_weights']}` |",
                f"| Attention Output | `[B, T, H]` | `{shapes['output']}` |",
            ]
        )
    return f"| Tensor | Shape | Measured |\n|---|---|---|\n{body}"


def _interpretation(result: dict) -> str:
    count = result.get("num_heads", 1)
    noun = "head" if count == 1 else "heads"
    width = count * result["head_dim"]
    if result["model"] == "multi_head_attention":
        opening = (
            f"Multi-head attention uses {count} {noun} of dimension {result['head_dim']} "
            f"(total attention width {width}) and has {result['parameter_count']:,} trainable parameters."
        )
    else:
        opening = (
            f"This causal attention module uses {count} {noun} of dimension "
            f"{result['head_dim']} and has {result['parameter_count']:,} trainable parameters."
        )
    return (
        f"{opening} On this run the forward + backward step processed about "
        f"{_fmt_number(result['tokens_per_second'], 0)} tokens/sec, using a batch of "
        f"{result['batch_size']} sequences of length {result['sequence_length']}. {_memory_text(result)}\n\n"
        "These numbers describe compute, parameters, and memory for this module. "
        "They are not evidence about which architecture learns TinyStories better."
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
    title = result.get("report_title", "Single-Head Self-Attention Benchmark Report")
    head_count = mask.get("head_count", 1)
    mask_scope = (
        f"All {head_count} heads passed."
        if head_count > 1 and mask["passed"]
        else f"Heads checked: {head_count}. Status: {'passed' if mask['passed'] else 'failed'}."
    )
    future_table = ""
    if result["model"] == "single_head_attention":
        future_table = f"""
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

    return f"""# {title}

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

{_model_config_lines(result)}

## Computational Performance

| Metric | Result |
|---|---:|
| Trainable Parameters | {result["parameter_count"]:,} |
| Forward Latency | {result["forward_latency_ms"]:.3f} ms |
| Forward + Backward Latency | {result["forward_backward_latency_ms"]:.3f} ms |
| Throughput | {_fmt_number(result["tokens_per_second"], 0)} tokens/sec |
| Peak GPU Memory | {memory_cell} |

## Causal Mask Verification

Status: {"passed" if mask["passed"] else "failed"}. {mask_scope}

Future attention probabilities are {"approximately zero" if mask["future_probabilities_near_zero"] else "not approximately zero"}.

{position_lines}

Example attention weights for one sequence of length {mask["sequence_length"]}. When several heads are checked, this matrix is head 1 only. Rows are queries and columns are keys. Entries above the diagonal are the masked future positions.

| | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
{matrix_rows}

## Shape Verification

{_shape_table(shapes)}

## Interpretation

{_interpretation(result)}

## Limitations

This benchmark measures only this attention module. It does not measure full language-model quality.

- There is no complete Transformer block yet.
- There is no language-model head yet.
- Validation loss comparison is not yet available.
- Runtime measurements can vary with GPU state and other system load.
{future_table}"""


def save_benchmark(result: dict, json_path: Path = JSON_PATH, markdown_path: Path = MARKDOWN_PATH) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(generate_benchmark_report(result), encoding="utf-8")


def print_baseline_summary(result: dict, title: str = "=== Single-Head Attention Baseline ===") -> None:
    memory = (
        "unavailable"
        if result["peak_gpu_memory_mb"] is None
        else f"{result['peak_gpu_memory_mb']:.2f} MB"
    )
    print(title)
    print(f"Parameters:               {result['parameter_count']:,}")
    print(f"Forward latency:          {result['forward_latency_ms']:.3f} ms")
    print(f"Forward + backward:       {result['forward_backward_latency_ms']:.3f} ms")
    print(f"Throughput:               {result['tokens_per_second']:,.0f} tokens/sec")
    print(f"Peak GPU memory:          {memory}")


def _memory_cell(result: dict) -> str:
    if result["peak_gpu_memory_mb"] is None:
        return "unavailable"
    return f"{result['peak_gpu_memory_mb']:.2f} MB"


def _delta(left: float, right: float, digits: int, suffix: str = "") -> str:
    value = left - right
    if digits == 0:
        return f"{value:+,.0f}{suffix}"
    return f"{value:+,.{digits}f}{suffix}"


def generate_attention_comparison(single: dict, multi: dict, single_wide: dict) -> str:
    """Build the comparison report from saved benchmark dictionaries."""
    gpu_name = multi["gpu_name"] or "none"
    cuda_version = multi["cuda_version"] or "none"
    memory_delta = (
        "unavailable"
        if multi["peak_gpu_memory_mb"] is None or single["peak_gpu_memory_mb"] is None
        else _delta(multi["peak_gpu_memory_mb"], single["peak_gpu_memory_mb"], 2, " MB")
    )
    wide_memory = _memory_cell(single_wide)
    multi_memory = _memory_cell(multi)
    return f"""# Single-Head vs Multi-Head Attention

## Experimental Setup

- Device: `{multi["device"]}`
- GPU name: {gpu_name}
- PyTorch version: {multi["pytorch_version"]}
- CUDA version: {cuda_version}
- Batch size: {multi["batch_size"]}
- Sequence length: {multi["sequence_length"]}
- Embedding dimension: {multi["embedding_dim"]}
- Dropout: {multi["dropout"]}
- Warm-up iterations: {multi["warmup_iterations"]}
- Measured iterations: {multi["measured_iterations"]}
- Input shape for every run: `[{multi["batch_size"]}, {multi["sequence_length"]}, {multi["embedding_dim"]}]`
- Throughput is `(batch_size × sequence_length) / forward_backward_seconds`.
- Difference in the table below is Multi Head minus Single Head.

The single-head baseline uses `head_dim = {single["head_dim"]}`. Multi-head attention uses `{multi["num_heads"]} × {multi["head_dim"]} = {multi["num_heads"] * multi["head_dim"]}` total attention width. These two runs do not have equal representational capacity. A difference in speed or parameter count here is not evidence that one architecture learns better.

## Computational Performance

| Metric | Single Head | Multi Head | Difference |
|---|---:|---:|---:|
| Parameters | {single["parameter_count"]:,} | {multi["parameter_count"]:,} | {_delta(multi["parameter_count"], single["parameter_count"], 0)} |
| Forward Latency | {single["forward_latency_ms"]:.3f} ms | {multi["forward_latency_ms"]:.3f} ms | {_delta(multi["forward_latency_ms"], single["forward_latency_ms"], 3, " ms")} |
| Forward + Backward | {single["forward_backward_latency_ms"]:.3f} ms | {multi["forward_backward_latency_ms"]:.3f} ms | {_delta(multi["forward_backward_latency_ms"], single["forward_backward_latency_ms"], 3, " ms")} |
| Throughput | {_fmt_number(single["tokens_per_second"], 0)} tok/s | {_fmt_number(multi["tokens_per_second"], 0)} tok/s | {_delta(multi["tokens_per_second"], single["tokens_per_second"], 0, " tok/s")} |
| Peak GPU Memory | {_memory_cell(single)} | {_memory_cell(multi)} | {memory_delta} |

## Parameter Comparison

Single Head 32 has {single["parameter_count"]:,} trainable parameters. Multi Head has {multi["parameter_count"]:,}. The extra parameters come from three more heads plus the output projection. Parameter count is not a quality score.

## Throughput and Latency

On this run, one forward pass took {single["forward_latency_ms"]:.3f} ms for Single Head 32 and {multi["forward_latency_ms"]:.3f} ms for Multi Head. Forward + backward took {single["forward_backward_latency_ms"]:.3f} ms and {multi["forward_backward_latency_ms"]:.3f} ms. Throughput was {_fmt_number(single["tokens_per_second"], 0)} tok/s and {_fmt_number(multi["tokens_per_second"], 0)} tok/s.

## GPU Memory

Peak allocated memory during the timed forward + backward steps was {_memory_cell(single)} for Single Head 32 and {multi_memory} for Multi Head.

## Architectural Difference

Single-head attention learns one attention pattern space. Multi-head attention runs several heads on the same input, concatenates their outputs, and mixes them with a linear projection.

After training, different heads may specialize. Examples of possible specializations, not patterns observed in this untrained benchmark:

- Head 1 may focus on local token relationships.
- Head 2 may focus on subject-verb relationships.
- Head 3 may focus on long-range dependencies.
- Head 4 may focus on other contextual patterns.

This benchmark does not show that any head has learned those roles.

## Interpretation

The first table compares a 32-wide head with four 32-wide heads. Multi-head attention has more parameters and a wider combined attention representation, so a latency or memory gap is expected and does not decide which design is better.

## Capacity-Controlled Comparison

Single Head 128 is one head with `head_dim = 128`. Multi Head is `{multi["num_heads"]}` heads with `head_dim = {multi["head_dim"]}`. Both have total attention width 128, and both read the same `[B, T, C]` input. This is the more appropriate comparison for the architecture itself. It is still only a compute and capacity comparison. Multi-head attention also has an output projection, so the parameter counts are not forced to match. In this code each head is a separate module and runs one after another, so the latency includes that sequential loop. A later fused implementation could spend its time differently.

| Metric | Single Head 128 | Multi Head 4×32 |
|---|---:|---:|
| Total Attention Width | 128 | 128 |
| Parameters | {single_wide["parameter_count"]:,} | {multi["parameter_count"]:,} |
| Forward Latency | {single_wide["forward_latency_ms"]:.3f} ms | {multi["forward_latency_ms"]:.3f} ms |
| Forward + Backward | {single_wide["forward_backward_latency_ms"]:.3f} ms | {multi["forward_backward_latency_ms"]:.3f} ms |
| Throughput | {_fmt_number(single_wide["tokens_per_second"], 0)} tok/s | {_fmt_number(multi["tokens_per_second"], 0)} tok/s |
| Peak GPU Memory | {wide_memory} | {multi_memory} |

## Limitations

Computational performance is not model quality. This comparison does not include training loss or validation loss.

- There is no complete Transformer block yet.
- There is no language-model head yet.
- There is no loss or training loop yet.
- Runtime measurements can vary with GPU state and other system load.

## Next Experiment

Add a Transformer block: residual connections, layer normalization, and a feed-forward network. After that, add a language-model head, loss, and a fixed-step training loop so single-head and multi-head models can be compared on validation loss under the same data, batch size, block size, embedding size, optimizer, and learning rate.
"""


def load_benchmark(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_parameter_breakdown(block: nn.Module) -> dict:
    """Count trainable parameters in each part of one Transformer block."""

    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    attention = count(block.attention)
    ffn = count(block.ffn)
    layernorm = count(block.ln1) + count(block.ln2)
    total = count(block)
    if attention + ffn + layernorm != total:
        raise RuntimeError("Transformer block parameter groups do not add up to the total.")
    return {
        "attention_parameters": attention,
        "ffn_parameters": ffn,
        "layernorm_parameters": layernorm,
        "total_parameters": total,
    }


def generate_transformer_block_report(result: dict) -> str:
    """Render the Transformer block report from one benchmark dictionary."""
    shapes = result["shapes"]
    rows = "\n".join(
        f"| {item['stage']} | `{item['symbolic']}` | `{item['measured']}` |"
        for item in shapes
    )
    params = result["parameters"]
    ffn_share = 100.0 * params["ffn_parameters"] / params["total_parameters"]
    gpu_name = result["gpu_name"] or "none"
    cuda_version = result["cuda_version"] or "none"
    memory = _memory_cell(result)
    return f"""# Transformer Block Benchmark Report

## Architecture

Pre-Norm Transformer block. Each sublayer is normalized first, then added back to its input.

```text
X
│
├── LayerNorm
│      ↓
│   Multi-Head Attention
│      ↓
└──── + Residual
       ↓
       X
       │
       ├── LayerNorm
       │      ↓
       │     FFN
       │      ↓
       └──── + Residual
              ↓
            Output
```

Attention moves information across tokens. The feed-forward network transforms each token vector on its own.

## Configuration

- Device: `{result["device"]}`
- GPU name: {gpu_name}
- PyTorch version: {result["pytorch_version"]}
- CUDA version: {cuda_version}
- Batch size: {result["batch_size"]}
- Sequence length: {result["sequence_length"]}
- Embedding dimension: {result["embedding_dim"]}
- Heads: {result["num_heads"]}
- Dropout: {result["dropout"]}
- Warm-up iterations: {result["warmup_iterations"]}
- Measured iterations: {result["measured_iterations"]}
- Output finite: {result["output_finite"]}
- Gradients present: {result["gradients_present"]}

## Shape Flow

| Stage | Shape | Measured |
|---|---|---|
{rows}

## Parameter Breakdown

| Part | Parameters |
|---|---:|
| Attention | {params["attention_parameters"]:,} |
| Feed Forward | {params["ffn_parameters"]:,} |
| LayerNorm | {params["layernorm_parameters"]:,} |
| Total | {params["total_parameters"]:,} |

The feed-forward network holds {ffn_share:.1f}% of the block's trainable parameters.

## Computational Performance

| Metric | Result |
|---|---:|
| Trainable Parameters | {params["total_parameters"]:,} |
| Forward Latency | {result["forward_latency_ms"]:.3f} ms |
| Forward + Backward Latency | {result["forward_backward_latency_ms"]:.3f} ms |
| Throughput | {_fmt_number(result["tokens_per_second"], 0)} tokens/sec |
| Peak GPU Memory | {memory} |

## Interpretation

One block preserves the input shape and contains {params["total_parameters"]:,} trainable parameters. On this run a forward + backward step processed about {_fmt_number(result["tokens_per_second"], 0)} tokens/sec. Peak GPU memory was {memory}. Most of the parameters sit in the feed-forward expansion from {result["embedding_dim"]} to {4 * result["embedding_dim"]} and back.

This is a compute and shape check for a single block. It is not a trained language model.

## Limitations

- Only one Transformer block is measured.
- Blocks are not stacked yet.
- There is no final LayerNorm, language-model head, loss, or training loop.
- Runtime measurements can vary with GPU state and other system load.

## Next Step

Stack several copies of this block. After the stack, add a final LayerNorm and a language-model head, then train with cross-entropy loss.
"""


def minigpt_parameter_breakdown(model: nn.Module) -> dict:
    """Count trainable parameters in each part of the full forward model."""

    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    groups = {
        "token_embedding_parameters": count(model.token_embedding),
        "position_embedding_parameters": count(model.position_embedding),
        "transformer_block_parameters": count(model.blocks),
        "final_layernorm_parameters": count(model.final_ln),
        "lm_head_parameters": count(model.lm_head),
    }
    total = count(model)
    if sum(groups.values()) != total:
        raise RuntimeError("MiniGPT parameter groups do not add up to the total.")
    groups["total_parameters"] = total
    groups["total_parameters_millions"] = total / 1_000_000
    return groups


def generate_minigpt_forward_report(result: dict) -> str:
    """Render the full-model forward report from one benchmark dictionary."""
    shapes = result["shapes"]
    rows = "\n".join(
        f"| {item['stage']} | `{item['symbolic']}` | `{item['measured']}` |"
        for item in shapes
    )
    params = result["parameter_groups"]
    gpu_name = result["gpu_name"] or "none"
    cuda_version = result["cuda_version"] or "none"
    memory = _memory_cell(result)
    millions = params["total_parameters_millions"]
    return f"""# MiniGPT Forward Architecture Report

## Architecture

```text
{result["architecture_summary"]}
```

Token ids become vectors, pass through {result["num_layers"]} Transformer blocks, and the LM head scores every vocabulary token. The forward pass returns raw logits. It does not apply softmax.

## Configuration

- Device: `{result["device"]}`
- GPU name: {gpu_name}
- PyTorch version: {result["pytorch_version"]}
- CUDA version: {cuda_version}
- Batch size: {result["batch_size"]}
- Sequence length: {result["sequence_length"]}
- Embedding dimension: {result["embedding_dim"]}
- Heads: {result["num_heads"]}
- Layers: {result["num_layers"]}
- Vocabulary size: {result["vocab_size"]}
- Dropout: {result["dropout"]}
- Warm-up iterations: {result["warmup_iterations"]}
- Measured iterations: {result["measured_iterations"]}
- Logits finite: {result["logits_finite"]}
- Gradients present: {result["gradients_present"]}

## Shape Flow

| Stage | Shape | Measured |
|---|---|---|
{rows}

## Parameter Breakdown

| Part | Parameters |
|---|---:|
| Token Embedding | {params["token_embedding_parameters"]:,} |
| Position Embedding | {params["position_embedding_parameters"]:,} |
| Transformer Blocks | {params["transformer_block_parameters"]:,} |
| Final LayerNorm | {params["final_layernorm_parameters"]:,} |
| LM Head | {params["lm_head_parameters"]:,} |
| Total | {params["total_parameters"]:,} |

{params["total_parameters"]:,} parameters ({millions:.2f}M).

Adding one Transformer block adds the same block parameter count again. Embeddings, the final LayerNorm, and the LM head stay fixed.

## Computational Performance

| Metric | Result |
|---|---:|
| Trainable Parameters | {params["total_parameters"]:,} |
| Forward Latency | {result["forward_latency_ms"]:.3f} ms |
| Forward + Backward Latency | {result["forward_backward_latency_ms"]:.3f} ms |
| Throughput | {_fmt_number(result["tokens_per_second"], 0)} tokens/sec |
| Peak GPU Memory | {memory} |

## Logits Interpretation

`logits[b, t, :]` holds one raw score for every vocabulary token. `logits[0, 10, :]` is the model's next-token scores for sequence 0 at position 10. These are not probabilities. CrossEntropyLoss applies log-softmax internally, together with negative log likelihood, and that combination is more stable on raw logits.

## Gradient Verification

A dummy objective, `logits.mean()`, produced finite gradients for the token embedding, an attention projection, a feed-forward weight, the final LayerNorm, and the LM head. This checks that gradients can flow. It is not language-model training.

## Limitations

- No cross-entropy loss against the next token is computed.
- There is no optimizer, training loop, validation loop, or checkpointing.
- Text generation is not implemented.
- Runtime measurements can vary with GPU state and other system load.

## Next Step

Train with cross-entropy on the shifted targets from `TextDataset`. The loss should compare `logits[:, :-1, :]` with the next token ids, or use the existing `(x, y)` pairs directly.
"""
