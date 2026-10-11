import json

import torch
from torch.utils.data import DataLoader

import config
from src.benchmark import (
    BLOCK_JSON_PATH,
    BLOCK_MARKDOWN_PATH,
    COMPARISON_CONFIG,
    COMPARISON_MARKDOWN_PATH,
    JSON_PATH,
    MARKDOWN_PATH,
    MULTI_JSON_PATH,
    MULTI_MARKDOWN_PATH,
    WIDE_JSON_PATH,
    WIDE_MARKDOWN_PATH,
    assemble_result,
    block_parameter_breakdown,
    device_metadata,
    generate_attention_comparison,
    generate_transformer_block_report,
    load_benchmark,
    measure_module,
    print_baseline_summary,
    save_benchmark,
    shapes_from_details,
    verify_causal_mask,
)
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.model import MiniGPT, MultiHeadAttention, SelfAttentionHead, TransformerBlock
from src.tokenizer import Tokenizer


def encode_validation(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode validation text with the training vocabulary.

    Unknown characters are reported. They are not dropped or replaced,
    because validation must use the same token ids as training.
    """
    try:
        return tokenizer.encode(text)
    except ValueError as exc:
        raise ValueError(
            "Validation text contains characters that are not in the training vocabulary. "
            "The tokenizer is fit only on training stories, so those characters have no token id.\n"
            f"{exc}"
        ) from exc


def main():
    train_stories, valid_stories = load_tinystories(num_train=5000, num_valid=500)
    train_text = prepare_text(train_stories)
    valid_text = prepare_text(valid_stories)

    # Fit on training text only, then reuse that vocabulary for validation.
    tokenizer = Tokenizer()
    tokenizer.fit(train_text)
    tokenizer.save(config.TOKENIZER_PATH)

    train_ids = tokenizer.encode(train_text)
    valid_ids = encode_validation(tokenizer, valid_text)

    train_dataset = TextDataset(train_ids, block_size=config.BLOCK_SIZE)
    valid_dataset = TextDataset(valid_ids, block_size=config.BLOCK_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    train_x, train_y = next(iter(train_loader))
    valid_x, valid_y = next(iter(valid_loader))

    # First window, not a shuffled batch, so the one-character shift is easy to see.
    sample_x, sample_y = train_dataset[0]

    print("training stories:", len(train_stories))
    print("validation stories:", len(valid_stories))
    print("vocab size:", tokenizer.vocab_size)
    print("training token ids:", len(train_ids))
    print("validation token ids:", len(valid_ids))
    print("train batch x:", tuple(train_x.shape))
    print("train batch y:", tuple(train_y.shape))
    print("valid batch x:", tuple(valid_x.shape))
    print("valid batch y:", tuple(valid_y.shape))
    print()
    print("Input:")
    print(tokenizer.decode(sample_x.tolist()))
    print()
    print("Target:")
    print(tokenizer.decode(sample_y.tolist()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.VOCAB_SIZE = tokenizer.vocab_size
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.BLOCK_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
    ).to(device)
    train_x = train_x.to(device)

    positions = torch.arange(train_x.shape[1], device=train_x.device)
    token_emb = model.token_embedding(train_x)
    position_emb = model.position_embedding(positions)
    output = model(train_x)

    print()
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("Token IDs shape:", tuple(train_x.shape))
    print("Token embedding shape:", tuple(token_emb.shape))
    print("Position embedding shape:", tuple(position_emb.shape))
    print("Final embedding output shape:", tuple(output.shape))
    print("Output device:", output.device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    if torch.cuda.is_available() and output.device.type != "cuda":
        raise RuntimeError(f"Expected the embedding output on CUDA, got {output.device}.")

    check_self_attention(train_x, output, torch.device(device))
    check_multi_head_attention(output, torch.device(device))
    check_transformer_block(output, torch.device(device))


def check_self_attention(token_ids: torch.Tensor, embedded: torch.Tensor, device: torch.device):
    """Run one causal head on a real batch, then record the comparison baseline."""
    settings = COMPARISON_CONFIG
    head = SelfAttentionHead(
        embedding_dim=settings["embedding_dim"],
        head_dim=settings["head_dim"],
        block_size=settings["sequence_length"],
        dropout=settings["dropout"],
    ).to(device)

    attention_output, details = head(embedded, return_details=True)
    if torch.cuda.is_available() and attention_output.device.type != "cuda":
        raise RuntimeError(f"Expected attention output on CUDA, got {attention_output.device}.")

    print()
    print("Token IDs:             ", tuple(token_ids.shape))
    print("Embedded X:            ", tuple(embedded.shape))
    print("Q:                     ", tuple(details["q"].shape))
    print("K:                     ", tuple(details["k"].shape))
    print("V:                     ", tuple(details["v"].shape))
    print("Attention scores:      ", tuple(details["attention_scores"].shape))
    print("Attention weights:     ", tuple(details["attention_weights"].shape))
    print("Attention output:      ", tuple(attention_output.shape))

    causal = verify_causal_mask(head, settings["embedding_dim"], device)
    print()
    print("Causal mask passed:", causal["passed"])
    print("Attention weights for T = 4:")
    for row in causal["weight_matrix"]:
        print(" ", row)
    if not causal["passed"]:
        raise RuntimeError("Causal mask verification failed.")

    measured = measure_module(
        head,
        batch_size=settings["batch_size"],
        sequence_length=settings["sequence_length"],
        embedding_dim=settings["embedding_dim"],
        warmup_iterations=settings["warmup_iterations"],
        measured_iterations=settings["measured_iterations"],
        device=device,
    )
    result = assemble_result(
        measured,
        shapes_from_details(token_ids, embedded, details),
        causal,
        device_metadata(device),
    )
    save_benchmark(result)
    print()
    print_baseline_summary(result)
    print()
    print("Saved benchmark JSON:")
    print(JSON_PATH.as_posix())
    print()
    print("Saved benchmark report:")
    print(MARKDOWN_PATH.as_posix())


def check_multi_head_attention(embedded: torch.Tensor, device: torch.device):
    """Run four causal heads on one batch, then compare compute with both single-head widths."""
    settings = COMPARISON_CONFIG
    multi = MultiHeadAttention(
        embedding_dim=settings["embedding_dim"],
        num_heads=settings["num_heads"],
        block_size=settings["sequence_length"],
        dropout=settings["dropout"],
    ).to(device)
    multi.eval()
    with torch.no_grad():
        projected, details = multi(embedded, return_details=True)

    print()
    print("Input X:              ", tuple(embedded.shape))
    for index, head_output in enumerate(details["head_outputs"], start=1):
        print(f"Head {index} output:       ", tuple(head_output.shape))
    print("Concatenated output:  ", tuple(details["concatenated"].shape))
    print("Projected output:     ", tuple(projected.shape))

    expected_heads = (
        embedded.shape[0],
        embedded.shape[1],
        settings["head_dim"],
    )
    if tuple(projected.shape) != tuple(embedded.shape):
        raise RuntimeError(f"Expected projected shape {tuple(embedded.shape)}, got {tuple(projected.shape)}.")
    for head_output in details["head_outputs"]:
        if tuple(head_output.shape) != expected_heads:
            raise RuntimeError(f"Expected head shape {expected_heads}, got {tuple(head_output.shape)}.")
    if torch.cuda.is_available() and projected.device.type != "cuda":
        raise RuntimeError(f"Expected multi-head output on CUDA, got {projected.device}.")

    causal = verify_causal_mask(multi, settings["embedding_dim"], device)
    print("Causal mask passed for all heads:", causal["passed"])
    if not causal["passed"]:
        raise RuntimeError("Multi-head causal mask verification failed.")

    measured = measure_module(
        multi,
        batch_size=settings["batch_size"],
        sequence_length=settings["sequence_length"],
        embedding_dim=settings["embedding_dim"],
        warmup_iterations=settings["warmup_iterations"],
        measured_iterations=settings["measured_iterations"],
        device=device,
    )
    multi_shapes = {
        "input": list(embedded.shape),
        "head_outputs": [list(head_output.shape) for head_output in details["head_outputs"]],
        "concatenated": list(details["concatenated"].shape),
        "output": list(projected.shape),
    }
    multi_result = assemble_result(
        measured,
        multi_shapes,
        causal,
        device_metadata(device),
        model="multi_head_attention",
        head_dim=settings["head_dim"],
        num_heads=settings["num_heads"],
        report_title="Multi-Head Self-Attention Benchmark Report",
    )
    save_benchmark(multi_result, MULTI_JSON_PATH, MULTI_MARKDOWN_PATH)

    wide = SelfAttentionHead(
        embedding_dim=settings["embedding_dim"],
        head_dim=settings["embedding_dim"],
        block_size=settings["sequence_length"],
        dropout=settings["dropout"],
    ).to(device)
    wide.eval()
    with torch.no_grad():
        wide_output, wide_details = wide(embedded, return_details=True)
    if tuple(wide_output.shape) != tuple(embedded.shape):
        raise RuntimeError(
            f"Expected Single Head 128 output {tuple(embedded.shape)}, got {tuple(wide_output.shape)}."
        )
    wide_causal = verify_causal_mask(wide, settings["embedding_dim"], device)
    if not wide_causal["passed"]:
        raise RuntimeError("Single Head 128 causal mask verification failed.")
    wide_measured = measure_module(
        wide,
        batch_size=settings["batch_size"],
        sequence_length=settings["sequence_length"],
        embedding_dim=settings["embedding_dim"],
        warmup_iterations=settings["warmup_iterations"],
        measured_iterations=settings["measured_iterations"],
        device=device,
    )
    wide_result = assemble_result(
        wide_measured,
        shapes_from_details(embedded[:, :, 0], embedded, wide_details),
        wide_causal,
        device_metadata(device),
        model="single_head_128",
        head_dim=settings["embedding_dim"],
        num_heads=1,
        report_title="Single-Head 128 Benchmark Report",
    )
    save_benchmark(wide_result, WIDE_JSON_PATH, WIDE_MARKDOWN_PATH)

    comparison = generate_attention_comparison(load_benchmark(JSON_PATH), multi_result, wide_result)
    COMPARISON_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_MARKDOWN_PATH.write_text(comparison, encoding="utf-8")

    print()
    print_baseline_summary(multi_result, "=== Multi-Head Attention Baseline ===")
    print()
    print_baseline_summary(wide_result, "=== Single Head 128 Baseline ===")
    print()
    print("Saved benchmark JSON:")
    print(MULTI_JSON_PATH.as_posix())
    print(WIDE_JSON_PATH.as_posix())
    print()
    print("Saved benchmark report:")
    print(MULTI_MARKDOWN_PATH.as_posix())
    print(WIDE_MARKDOWN_PATH.as_posix())
    print(COMPARISON_MARKDOWN_PATH.as_posix())


def check_transformer_block(embedded: torch.Tensor, device: torch.device):
    """Check one Pre-Norm block on a real batch, then record its compute baseline."""
    settings = COMPARISON_CONFIG
    block = TransformerBlock(
        embedding_dim=settings["embedding_dim"],
        num_heads=settings["num_heads"],
        block_size=settings["sequence_length"],
        dropout=settings["dropout"],
    ).to(device)

    block.eval()
    with torch.no_grad():
        output, details = block(embedded, return_details=True)

    print()
    print("Input shape:         ", tuple(details["input"].shape))
    print("After LayerNorm 1:   ", tuple(details["layernorm_1"].shape))
    print("After Attention:     ", tuple(details["attention"].shape))
    print("After Residual 1:    ", tuple(details["residual_1"].shape))
    print("After LayerNorm 2:   ", tuple(details["layernorm_2"].shape))
    print("After FFN:           ", tuple(details["ffn"].shape))
    print("After Residual 2:    ", tuple(details["output"].shape))
    print("Final output shape:  ", tuple(output.shape))

    if tuple(output.shape) != tuple(embedded.shape):
        raise RuntimeError(
            f"Expected Transformer block output {tuple(embedded.shape)}, got {tuple(output.shape)}."
        )
    for name, tensor in details.items():
        if tuple(tensor.shape) != tuple(embedded.shape):
            raise RuntimeError(f"{name} changed shape to {tuple(tensor.shape)}.")
    if not torch.isfinite(output).all():
        raise RuntimeError("Transformer block output contains NaN or infinity.")
    if torch.cuda.is_available() and output.device.type != "cuda":
        raise RuntimeError(f"Expected Transformer block output on CUDA, got {output.device}.")

    block.zero_grad(set_to_none=True)
    block.train()
    trained = block(embedded.detach())
    if not torch.isfinite(trained).all():
        raise RuntimeError("Transformer block training forward contains NaN or infinity.")
    trained.mean().backward()
    watched = {
        "attention projection weights": block.attention.output_projection.weight.grad,
        "FFN expand weights": block.ffn.expand.weight.grad,
        "FFN project weights": block.ffn.project.weight.grad,
        "LayerNorm 1 weight": block.ln1.weight.grad,
        "LayerNorm 2 weight": block.ln2.weight.grad,
    }
    for name, grad in watched.items():
        if grad is None or not torch.isfinite(grad).all() or torch.count_nonzero(grad) == 0:
            raise RuntimeError(f"{name} did not receive a finite gradient.")

    params = block_parameter_breakdown(block)
    print("Attention parameters:     ", params["attention_parameters"])
    print("FFN parameters:           ", params["ffn_parameters"])
    print("LayerNorm parameters:     ", params["layernorm_parameters"])
    print("Total block parameters:   ", params["total_parameters"])

    measured = measure_module(
        block,
        batch_size=settings["batch_size"],
        sequence_length=settings["sequence_length"],
        embedding_dim=settings["embedding_dim"],
        warmup_iterations=settings["warmup_iterations"],
        measured_iterations=settings["measured_iterations"],
        device=device,
    )
    if measured["parameter_count"] != params["total_parameters"]:
        raise RuntimeError("Benchmark parameter count does not match the block breakdown.")

    stage_names = [
        ("Input", details["input"]),
        ("LayerNorm 1", details["layernorm_1"]),
        ("Multi-Head Attention", details["attention"]),
        ("Residual 1", details["residual_1"]),
        ("LayerNorm 2", details["layernorm_2"]),
        ("Feed Forward", details["ffn"]),
        ("Final Output", details["output"]),
    ]
    result = {
        "model": "transformer_block",
        **device_metadata(device),
        "batch_size": settings["batch_size"],
        "sequence_length": settings["sequence_length"],
        "embedding_dim": settings["embedding_dim"],
        "num_heads": settings["num_heads"],
        "dropout": settings["dropout"],
        "warmup_iterations": settings["warmup_iterations"],
        "measured_iterations": settings["measured_iterations"],
        **measured,
        "parameters": params,
        "shapes": [
            {"stage": stage, "symbolic": "[B, T, C]", "measured": list(tensor.shape)}
            for stage, tensor in stage_names
        ],
        "output_finite": True,
        "gradients_present": True,
    }
    BLOCK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLOCK_JSON_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    BLOCK_MARKDOWN_PATH.write_text(generate_transformer_block_report(result), encoding="utf-8")

    print()
    print_baseline_summary(result, "=== Transformer Block Baseline ===")
    print()
    print("Saved benchmark JSON:")
    print(BLOCK_JSON_PATH.as_posix())
    print()
    print("Saved benchmark report:")
    print(BLOCK_MARKDOWN_PATH.as_posix())


if __name__ == "__main__":
    main()
