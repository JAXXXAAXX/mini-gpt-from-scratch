"""Measure whether the trained v2.1 model benefits from context beyond 64 tokens.

One checkpoint is evaluated three times. The architecture stays at block_size 256.
Each condition changes only how many previous tokens are visible for the same targets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import config
from evaluate_story import (
    contains_term,
    extract_memory_terms,
    run_memory_probe,
    select_memory_items,
)
from generate import generate_from_ids, load_model
from src.bpe_tokenizer import BPETokenizer
from src.data_utils import load_tinystories


def metric_change(before: float | None, after: float | None) -> dict:
    if before is None or after is None:
        return {"absolute": None, "relative_percent": None}
    absolute = after - before
    relative = None if before == 0 else 100.0 * absolute / abs(before)
    return {"absolute": absolute, "relative_percent": relative}


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def encode_validation_stories(stories: list[str], tokenizer: BPETokenizer) -> dict[int, list[int]]:
    encoded = {}
    for story_index, story in enumerate(stories):
        text = story.strip()
        if not text:
            continue
        encoded[story_index] = tokenizer.encode(text, add_bos=True, add_eos=True)
    return encoded


def eligible_targets(encoded: dict[int, list[int]], history: int) -> list[dict]:
    """Positions that have `history` previous tokens inside the same story."""
    targets = []
    for story_index, token_ids in encoded.items():
        for target_index in range(history, len(token_ids)):
            targets.append(
                {
                    "story_index": story_index,
                    "target_index": target_index,
                }
            )
    return targets


def choose_targets(targets: list[dict], count: int, seed: int) -> list[dict]:
    if len(targets) <= count:
        return targets
    generator = torch.Generator()
    generator.manual_seed(seed)
    picked = torch.randperm(len(targets), generator=generator)[:count]
    order = sorted(int(index) for index in picked.tolist())
    return [targets[index] for index in order]


def last_token_loss(
    model,
    encoded: dict[int, list[int]],
    targets: list[dict],
    visible: int,
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Cross-entropy of one shared target token under a cropped history."""
    sequences = []
    labels = []
    for item in targets:
        token_ids = encoded[item["story_index"]]
        target_index = item["target_index"]
        sequences.append(token_ids[target_index - visible : target_index])
        labels.append(token_ids[target_index])
    total = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_x = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
            batch_y = torch.tensor(labels[start : start + batch_size], dtype=torch.long, device=device)
            logits = model(batch_x)[:, -1, :]
            loss = F.cross_entropy(logits, batch_y, reduction="sum")
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite loss at visible context {visible}.")
            total += value
            seen += int(batch_y.shape[0])
    if seen == 0:
        raise RuntimeError(f"No targets were evaluated at visible context {visible}.")
    mean_loss = total / seen
    return {
        "visible_context": visible,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss <= 20 else float("inf"),
        "evaluated_tokens": seen,
        "evaluated_examples": seen,
    }


def select_story_examples(
    encoded: dict[int, list[int]],
    tokenizer: BPETokenizer,
    count: int,
    prefix_len: int,
) -> list[dict]:
    examples = []
    for story_index in sorted(encoded):
        token_ids = encoded[story_index]
        if len(token_ids) < prefix_len:
            continue
        prefix_ids = token_ids[:prefix_len]
        examples.append(
            {
                "story_index": story_index,
                "token_offset": 0,
                "prefix_len": prefix_len,
                "prefix_ids": prefix_ids,
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
            }
        )
        if len(examples) == count:
            break
    if len(examples) < count:
        raise RuntimeError(
            f"Needed {count} validation stories with at least {prefix_len} BPE tokens, "
            f"found {len(examples)}."
        )
    return examples


def generate_visible_continuations(model, tokenizer, example: dict, device: torch.device) -> list[dict]:
    generations = []
    prefix_ids = example["prefix_ids"]
    for visible in config.CONTEXT_UTIL_VISIBLE:
        visible_ids = prefix_ids[-visible:]
        prompt_text = tokenizer.decode(visible_ids, skip_special_tokens=True)
        hidden_ids = prefix_ids[: len(prefix_ids) - visible]
        hidden_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
        sample = generate_from_ids(
            model,
            tokenizer,
            prompt_ids=visible_ids,
            prompt_text=prompt_text,
            max_new_tokens=config.CONTEXT_UTIL_EXAMPLE_NEW_TOKENS,
            temperature=config.STORY_EVAL_TEMPERATURE,
            top_k=config.STORY_EVAL_TOP_K,
            seed=config.STORY_EVAL_SEED,
            device=device,
        )
        early_terms = extract_memory_terms(hidden_text)
        retained = [
            term["term"]
            for term in early_terms
            if contains_term(sample["continuation_text"], term["term"])
        ]
        generations.append(
            {
                "visible_context": visible,
                "hidden_text": hidden_text,
                "early_terms": early_terms,
                "retained_terms": retained,
                "prompt_text": prompt_text,
                "continuation_text": sample["continuation_text"],
                "stopped_on_eos": sample["stopped_on_eos"],
                "tokens_generated": sample["tokens_generated"],
            }
        )
    return generations


def _loss_step_sentence(left: int, right: int, before: float, after: float) -> str:
    drop = before - after
    if drop >= 0.05:
        return (
            f"From {left} to {right} visible tokens, loss falls by {drop:.4f}. "
            "That step improves prediction."
        )
    if drop >= 0.01:
        return (
            f"From {left} to {right} visible tokens, loss falls by {drop:.4f}. "
            "That step helps modestly."
        )
    if abs(drop) <= 0.001:
        return (
            f"From {left} to {right} visible tokens, loss is essentially unchanged "
            f"(change {after - before:+.4f})."
        )
    if drop > 0.001:
        return (
            f"From {left} to {right} visible tokens, loss falls by {drop:.4f}. "
            "That step is nearly flat."
        )
    return (
        f"From {left} to {right} visible tokens, loss rises by {-drop:.4f}. "
        "The extra history does not improve these predictions."
    )


def utilization_reading(by_visible: dict[int, dict], probes: dict[int, dict] | None = None) -> str:
    loss_64 = by_visible[64]["loss"]
    loss_128 = by_visible[128]["loss"]
    loss_256 = by_visible[256]["loss"]
    numbers = (
        f"Measured last-token loss is {loss_64:.4f} at 64 visible tokens, "
        f"{loss_128:.4f} at 128, and {loss_256:.4f} at 256."
    )
    steps = " ".join(
        [
            _loss_step_sentence(64, 128, loss_64, loss_128),
            _loss_step_sentence(128, 256, loss_128, loss_256),
            _loss_step_sentence(64, 256, loss_64, loss_256),
        ]
    )
    retention = ""
    if probes is not None:
        shown_64 = probes[64]["terms_retained"]
        total = probes[64]["terms_total"]
        shown_128 = probes[128]["terms_retained"]
        retention = (
            f" Early entity/content-word retention on the fixed 128-token probe is "
            f"{shown_64}/{total} with 64 visible tokens and {shown_128}/{total} with 128 visible tokens. "
            "Requesting 256 visible tokens cannot add history, because the probe prefix is only 128 tokens."
        )
    return (
        f"{numbers} {steps} The network can accept 256 tokens. "
        "These measurements are what show whether that extra history changes the prediction."
        f"{retention}"
    )


def build_report(result: dict) -> str:
    conditions = result["conditions"]
    rows = "\n".join(
        f"| {item['visible_context']} | {item['loss']:.4f} | {item['perplexity']:.2f} | "
        f"{item['evaluated_tokens']:,} |"
        for item in conditions
    )
    change_rows = "\n".join(
        f"| {item['metric']} | {item['from_context']} → {item['to_context']} | "
        f"{fmt(item['absolute'])} | {fmt(item['relative_percent'], 2)}% |"
        for item in result["changes"]
    )
    retention_rows = "\n".join(
        f"| {item['visible_tokens_requested']} | {item['visible_tokens_used']} | "
        f"{item['terms_retained']} / {item['terms_total']} | {fmt(item['retention_rate'], 3)} |"
        for item in result["memory_probe"]
    )
    example_sections = []
    for example in result["story_examples"]:
        blocks = [f"### Validation story {example['story_index']}\n"]
        blocks.append("Prefix available to the 256-token view:\n")
        blocks.append(f"```text\n{example['prefix_text']}\n```\n")
        for generation in example["generations"]:
            terms = ", ".join(term["term"] for term in generation["early_terms"][:12]) or "none"
            retained = ", ".join(generation["retained_terms"]) or "none"
            blocks.append(
                f"#### {generation['visible_context']} visible tokens\n\n"
                f"History hidden from this view:\n\n"
                f"```text\n{generation['hidden_text'] or '[none]'}\n```\n\n"
                f"Early entity/content words in that hidden span: {terms}\n\n"
                f"Continuation:\n\n"
                f"```text\n{generation['continuation_text']}\n```\n\n"
                f"Retained from the hidden span: {retained}. "
                f"Stopped on EOS: {generation['stopped_on_eos']}.\n"
            )
        example_sections.append("\n".join(blocks))
    probe_note = result["memory_probe_note"]
    label = result.get("label", "v2.1")
    return f"""# {label} Context Utilization

One trained checkpoint, `{result["checkpoint"]}`, is scored three times. The model architecture stays at block size 256. The tokenizer is the existing BPE file and is not refit. Each condition uses the same validation targets and changes only the number of previous tokens the model can see.

Architecture capacity means the network can accept 256 tokens. Learned utilization means loss or early-term retention actually changes when those extra tokens are shown.

## Validation Loss

Loss is the cross-entropy of one target token. The target is identical for 64, 128, and 256. Only the preceding history is cropped. This is not the average loss over every position in the window.

Seed {result['selection']['seed']} selected {result['selection']['selected_examples']} targets from {result['selection']['eligible_examples']} positions that have 256 tokens of same-story history.

| Visible Context | Validation Loss | Perplexity | Evaluated Tokens |
|---:|---:|---:|---:|
{rows}

## Changes

A negative loss or perplexity change means the longer view predicts the same targets better.

| Metric | Contexts | Absolute Change | Relative Change |
|---|---|---:|---:|
{change_rows}

## Interpretation

{result['interpretation']}

## Long-prefix Memory Probe

This reuses the existing probe. The prompt is the first 128 BPE tokens of a validation story. Tracked terms are entity and content words from the first 64 tokens that do not already appear in tokens 64–128. The same terms are scored at every visible length. This is a lexical memory proxy, not a linguistic noun tagger.

{probe_note}

| Requested Visible Tokens | Tokens Actually Shown | Matched Early Terms | Retention |
|---:|---:|---:|---:|
{retention_rows}

A lexical match can miss a paraphrase. If the prefix says "Toby carried his yellow ball" and the continuation says "The dog dropped his toy," Toby and ball are counted as missed even when the scene is still about that dog and that toy.

## Real Validation Stories

Decoding is temperature {config.STORY_EVAL_TEMPERATURE}, top-k {config.STORY_EVAL_TOP_K}, seed {config.STORY_EVAL_SEED}. Each continuation is generated by this same {label} checkpoint. The 64-token view receives only the end of the 256-token prefix.

{chr(10).join(example_sections)}
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score one checkpoint at visible context 64, 128, and 256.")
    parser.add_argument("--checkpoint", default=config.V2_1_CTX256_BEST_CHECKPOINT)
    parser.add_argument("--json", dest="json_path", default=config.CONTEXT_UTIL_JSON)
    parser.add_argument("--report", dest="report_path", default=config.CONTEXT_UTIL_REPORT)
    parser.add_argument("--reuse-targets", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. This benchmark does not train a model."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device.type)
    print("Checkpoint:", checkpoint_path)
    tokenizer = BPETokenizer.load(config.BPE_TOKENIZER_PATH)
    print("Loaded tokenizer:", config.BPE_TOKENIZER_PATH, "vocab", tokenizer.vocab_size)
    model, checkpoint = load_model(checkpoint_path, device)
    if model.block_size != 256:
        raise RuntimeError(
            f"v2.1 checkpoint block_size is {model.block_size}, expected 256."
        )

    _train, valid_stories = load_tinystories(
        num_train=1,
        num_valid=config.NUM_VALID_STORIES,
    )
    encoded = encode_validation_stories(valid_stories, tokenizer)
    pool = eligible_targets(encoded, history=max(config.CONTEXT_UTIL_VISIBLE))
    if args.reuse_targets:
        saved = json.loads(Path(args.reuse_targets).read_text(encoding="utf-8"))
        selected = saved["selection"]["targets"]
        print(f"Reusing {len(selected)} targets from {args.reuse_targets}.")
    else:
        selected = choose_targets(pool, config.CONTEXT_UTIL_SAMPLE_COUNT, config.CONTEXT_UTIL_SEED)
    print(f"Eligible targets: {len(pool)}. Selected: {len(selected)}.")

    conditions = []
    for visible in config.CONTEXT_UTIL_VISIBLE:
        print(f"Scoring visible context {visible}...")
        conditions.append(last_token_loss(model, encoded, selected, visible, device))
        print(
            f"  loss {conditions[-1]['loss']:.4f} "
            f"ppl {conditions[-1]['perplexity']:.2f} "
            f"tokens {conditions[-1]['evaluated_tokens']}"
        )
    by_visible = {item["visible_context"]: item for item in conditions}

    print("Selecting long-prefix probe stories...")
    memory_items = select_memory_items(valid_stories, tokenizer)
    gen_cfg = {
        "max_new_tokens": config.STORY_EVAL_MAX_NEW_TOKENS,
        "temperature": config.STORY_EVAL_TEMPERATURE,
        "top_k": config.STORY_EVAL_TOP_K,
        "seed": config.STORY_EVAL_SEED,
    }
    probes = []
    for visible in config.CONTEXT_UTIL_VISIBLE:
        print(f"Memory probe with {visible} requested visible tokens...")
        probe = run_memory_probe(
            model,
            tokenizer,
            memory_items,
            gen_cfg,
            device,
            visible_tokens=visible,
        )
        probes.append(
            {
                "visible_tokens_requested": visible,
                "visible_tokens_used": probe["visible_tokens_used"],
                "terms_retained": probe["terms_retained"],
                "terms_total": probe["terms_total"],
                "retention_rate": probe["retention_rate"],
                "story_count": probe["story_count"],
                "samples": [
                    {
                        "story_index": sample["story_index"],
                        "retained_terms": sample["retained_terms"],
                        "term_count": len(sample["memory_terms"]),
                        "continuation_text": sample.get("continuation_text", ""),
                        "stopped_on_eos": sample["stopping"]["stopped_naturally"],
                    }
                    for sample in probe["samples"]
                ],
            }
        )
        print(
            f"  retained {probe['terms_retained']}/{probe['terms_total']} "
            f"using {probe['visible_tokens_used']} tokens"
        )

    print("Generating real story examples...")
    examples = select_story_examples(
        encoded,
        tokenizer,
        config.CONTEXT_UTIL_EXAMPLE_COUNT,
        max(config.CONTEXT_UTIL_VISIBLE),
    )
    story_examples = []
    for example in examples:
        story_examples.append(
            {
                **{key: value for key, value in example.items() if key != "prefix_ids"},
                "generations": generate_visible_continuations(model, tokenizer, example, device),
            }
        )

    def changes_for(metric: str, reader) -> list[dict]:
        rows = []
        for left, right in ((64, 128), (128, 256), (64, 256)):
            change = metric_change(reader(by_visible[left]), reader(by_visible[right]))
            rows.append(
                {
                    "metric": metric,
                    "from_context": left,
                    "to_context": right,
                    **change,
                }
            )
        return rows

    probe_by_request = {item["visible_tokens_requested"]: item for item in probes}
    retention_changes = []
    for left, right in ((64, 128), (128, 256), (64, 256)):
        change = metric_change(
            probe_by_request[left]["retention_rate"],
            probe_by_request[right]["retention_rate"],
        )
        retention_changes.append(
            {
                "metric": "early entity/content-word retention",
                "from_context": left,
                "to_context": right,
                **change,
            }
        )

    used_256 = probe_by_request[256]["visible_tokens_used"]
    probe_note = (
        f"The probe prompt is {config.STORY_MEMORY_PREFIX_TOKENS} tokens. "
        f"A request for 256 visible tokens is capped at {used_256}, because no earlier "
        "tokens belong to this probe. The 128-token and 256-token probe conditions "
        "therefore show the same prefix. The benchmark definition is unchanged."
    )
    result = {
        "checkpoint": checkpoint_path,
        "label": args.label or "v2.1",
        "position_encoding": checkpoint.get("model_config", {}).get("position_encoding", "absolute"),
        "checkpoint_step": checkpoint.get("step"),
        "block_size": model.block_size,
        "tokenizer": config.BPE_TOKENIZER_PATH,
        "selection": {
            "seed": config.CONTEXT_UTIL_SEED,
            "eligible_examples": len(pool),
            "selected_examples": len(selected),
            "targets": selected,
        },
        "conditions": conditions,
        "changes": (
            changes_for("loss", lambda item: item["loss"])
            + changes_for("perplexity", lambda item: item["perplexity"])
            + retention_changes
        ),
        "interpretation": utilization_reading(by_visible, probe_by_request),
        "memory_probe": probes,
        "memory_probe_note": probe_note,
        "story_examples": story_examples,
        "decoding": {
            "temperature": config.STORY_EVAL_TEMPERATURE,
            "top_k": config.STORY_EVAL_TOP_K,
            "seed": config.STORY_EVAL_SEED,
        },
    }
    json_path = Path(args.json_path)
    report_path = Path(args.report_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(result), encoding="utf-8")
    print("Saved", json_path.as_posix())
    print("Saved", report_path.as_posix())
    print(result["interpretation"])


if __name__ == "__main__":
    main()
