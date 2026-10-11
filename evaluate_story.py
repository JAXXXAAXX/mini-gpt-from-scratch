"""Reusable TinyStories story-generation evaluation harness.

Works with character (v1) and BPE (v2) tokenizers selected by config.TOKENIZER_TYPE.
Does not retrain the model.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from evaluate import average_validation_loss, load_json, safe_print
from generate import generate_once, load_model
from src.data_utils import load_tinystories
from src.dataset import TextDataset
from src.tokenizer_utils import (
    encode_stories,
    has_eos,
    load_tokenizer,
    resolve_tokenizer_type,
)
from src.trainer import perplexity

# Permanent suite. Keep stable across v1 / v2 / v3 comparisons.
STORY_PROMPTS = [
    "Once there was",
    "One day a little girl",
    "The dog was very happy because",
    "In a small village",
    "The boy opened the door and",
    "A little cat wanted to",
    "The children went outside and",
    "There was a small house near",
]

MULTI_SEED_PROMPT = "Once there was"
MULTI_SEED_COUNT = 5
SENSITIVITY_PROMPT = "Once there was"


def words(text: str) -> list[str]:
    return [piece for piece in re.split(r"\s+", text.strip()) if piece]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens: list[str], n: int) -> float | None:
    grams = ngrams(tokens, n)
    if not grams:
        return None
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(tokens: list[str], n: int) -> float | None:
    """Share of n-gram occurrences that are repeats of an earlier identical n-gram."""
    grams = ngrams(tokens, n)
    if not grams:
        return None
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def max_repeated_phrase_count(tokens: list[str], sizes: tuple[int, ...] = (2, 3, 4)) -> int:
    best = 0
    for size in sizes:
        grams = ngrams(tokens, size)
        if grams:
            best = max(best, max(Counter(grams).values()))
    return best


def max_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    for index in range(1, len(text)):
        if text[index] == text[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def lexical_metrics(text: str) -> dict:
    token_list = words(text)
    unique = set(token_list)
    alpha = [token for token in token_list if any(ch.isalpha() for ch in token)]
    avg_len = sum(len(token) for token in token_list) / len(token_list) if token_list else 0.0
    return {
        "word_count": len(token_list),
        "unique_word_count": len(unique),
        "unique_word_ratio": len(unique) / len(token_list) if token_list else 0.0,
        "average_word_length": avg_len,
        "alphabetic_word_percentage": 100.0 * len(alpha) / len(token_list) if token_list else 0.0,
    }


def stopping_behavior(sample: dict, tokenizer) -> dict:
    text = sample["generated_text"]
    continuation = text[len(sample["prompt"]) :] if text.startswith(sample["prompt"]) else text
    truncated = bool(continuation) and continuation[-1] not in ".!?\n"
    eos_supported = bool(sample.get("eos_supported", has_eos(tokenizer)))
    stopped_on_eos = bool(sample.get("stopped_on_eos", False))
    hit_max = sample["tokens_generated"] >= sample["max_new_tokens"] and not stopped_on_eos
    return {
        "eos_supported": eos_supported,
        "stopped_naturally": stopped_on_eos,
        "hit_max_new_tokens": hit_max,
        "ended_in_obvious_truncation": truncated and not stopped_on_eos,
        "tokens_appended": sample["tokens_generated"],
    }


def story_diagnostics(sample: dict) -> dict:
    prompt = sample["prompt"]
    text = sample["generated_text"]
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    token_list = words(continuation)
    return {
        "continuation_character_count": len(continuation),
        "distinct_1": distinct_n(token_list, 1),
        "distinct_2": distinct_n(token_list, 2),
        "distinct_3": distinct_n(token_list, 3),
        "repeated_2gram_rate": repeated_ngram_rate(token_list, 2),
        "repeated_3gram_rate": repeated_ngram_rate(token_list, 3),
        "repeated_4gram_rate": repeated_ngram_rate(token_list, 4),
        "max_repeated_short_phrase_count": max_repeated_phrase_count(token_list),
        "max_repeated_character_run": max_char_run(continuation),
        "lexical": lexical_metrics(continuation),
    }


def empty_manual_rubric() -> dict:
    return {
        "coherence_score": None,
        "grammar_score": None,
        "story_progression_score": None,
        "repetition_score": None,
        "scale": {
            "0": "poor",
            "1": "partially acceptable",
            "2": "good",
        },
        "definitions": {
            "coherence": {
                "0": "unrelated or broken continuation",
                "1": "partially connected",
                "2": "clearly continues the prompt",
            },
            "grammar": {
                "0": "mostly malformed",
                "1": "understandable with frequent errors",
                "2": "mostly grammatical",
            },
            "story_progression": {
                "0": "no meaningful progression",
                "1": "some event development",
                "2": "clear continuation with events/actions",
            },
            "repetition": {
                "0": "severe repetition",
                "1": "noticeable repetition",
                "2": "little problematic repetition",
            },
        },
    }


def load_validation_stories(num_valid: int | None = None) -> tuple[list[str], dict]:
    """Load the same validation story count used for training."""
    num_valid = num_valid or config.NUM_VALID_STORIES
    print("Loading TinyStories validation stories via HF streaming slice...")
    _train, valid_stories = load_tinystories(num_train=1, num_valid=num_valid)
    meta = {
        "source": f"huggingface:roneneldan/TinyStories validation[:{num_valid}]",
        "valid_stories": len(valid_stories),
    }
    return valid_stories, meta


def annotate_sample(sample: dict, tokenizer: Tokenizer) -> dict:
    diagnostics = story_diagnostics(sample)
    stopping = stopping_behavior(sample, tokenizer)
    return {
        **sample,
        "diagnostics": diagnostics,
        "stopping": stopping,
        "manual_scores": {
            "coherence_score": None,
            "grammar_score": None,
            "story_progression_score": None,
            "repetition_score": None,
        },
    }


def mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(result: dict) -> str:
    model = result["model"]
    dataset = result["dataset"]
    validation = result["validation"]
    baselines = result.get("baselines") or {}
    config_gen = result["generation_config"]
    suite = result["story_suite"]
    multi = result["multi_seed_diversity"]
    sensitivity = result["sampling_sensitivity"]
    performance = result["performance"]
    baseline = result["v1_baseline_summary"]

    suite_sections = []
    for item in suite:
        diag = item["diagnostics"]
        lex = diag["lexical"]
        suite_sections.append(
            f"### Prompt\n\n"
            f"`{item['prompt']}`\n\n"
            f"### Generated continuation\n\n"
            f"```text\n{item['generated_text']}\n```\n\n"
            f"### Automatic diagnostics\n\n"
            f"| Metric | Result |\n"
            f"|---|---:|\n"
            f"| Distinct-1 | {fmt(diag['distinct_1'], 3)} |\n"
            f"| Distinct-2 | {fmt(diag['distinct_2'], 3)} |\n"
            f"| Distinct-3 | {fmt(diag['distinct_3'], 3)} |\n"
            f"| Repeated 2-gram Rate | {fmt(diag['repeated_2gram_rate'], 3)} |\n"
            f"| Repeated 3-gram Rate | {fmt(diag['repeated_3gram_rate'], 3)} |\n"
            f"| Repeated 4-gram Rate | {fmt(diag['repeated_4gram_rate'], 3)} |\n"
            f"| Max repeated short phrase count | {fmt(diag['max_repeated_short_phrase_count'], 0)} |\n"
            f"| Max repeated character run | {fmt(diag['max_repeated_character_run'], 0)} |\n"
            f"| Word Count | {fmt(lex['word_count'], 0)} |\n"
            f"| Unique Word Ratio | {fmt(lex['unique_word_ratio'], 3)} |\n"
            f"| Generation Speed | {fmt(item['tokens_per_second'], 1)} tokens/sec |\n"
            f"| Hit max_new_tokens | {item['stopping']['hit_max_new_tokens']} |\n"
            f"| Obvious truncation | {item['stopping']['ended_in_obvious_truncation']} |\n"
        )

    sensitivity_rows = "\n".join(
        f"| {item['temperature']} | "
        f"{'None' if item['top_k'] is None else item['top_k']} | "
        f"{fmt(item['diagnostics']['distinct_2'], 3)} | "
        f"{fmt(item['diagnostics']['repeated_3gram_rate'], 3)} | "
        f"{fmt(item['tokens_per_second'], 1)} |"
        for item in sensitivity
    )

    repetition_rows = "\n".join(
        f"| `{item['prompt']}` | "
        f"{fmt(item['diagnostics']['repeated_2gram_rate'], 3)} | "
        f"{fmt(item['diagnostics']['repeated_3gram_rate'], 3)} | "
        f"{fmt(item['diagnostics']['repeated_4gram_rate'], 3)} | "
        f"{fmt(item['diagnostics']['max_repeated_short_phrase_count'], 0)} |"
        for item in suite
    )

    diversity_rows = "\n".join(
        f"| `{item['prompt']}` | "
        f"{fmt(item['diagnostics']['distinct_1'], 3)} | "
        f"{fmt(item['diagnostics']['distinct_2'], 3)} | "
        f"{fmt(item['diagnostics']['distinct_3'], 3)} |"
        for item in suite
    )

    baseline_table = (
        f"| Uniform Random | {fmt(baselines.get('uniform_loss'))} | {fmt(baselines.get('uniform_perplexity'), 2)} |\n"
        f"| Untrained MiniGPT | {fmt(baselines.get('untrained_loss'))} | {fmt(baselines.get('untrained_perplexity'), 2)} |\n"
        f"| Trained MiniGPT | {fmt(validation['loss'])} | {fmt(validation['perplexity'], 2)} |"
    )

    version_label = "v2 BPE" if model.get("tokenizer_type") == "bpe" else "v1"
    return f"""# MiniGPT {version_label} Story Generation Evaluation

## 1. Model Configuration

| Setting | Value |
|---|---:|
| Tokenizer type | {model['tokenizer_type']} |
| Vocabulary size | {model['vocab_size']} |
| Block size | {model['block_size']} |
| Embedding dimension | {model['embedding_dim']} |
| Attention heads | {model['num_heads']} |
| Transformer layers | {model['num_layers']} |
| Dropout | {model['dropout']} |
| Parameters | {model['parameter_count']:,} |
| Checkpoint | `{model['checkpoint_path']}` |
| Training step | {fmt(model.get('checkpoint_step'), 0)} |

Primary generation settings:

| Setting | Value |
|---|---:|
| max_new_tokens | {config_gen['max_new_tokens']} |
| temperature | {config_gen['temperature']} |
| top_k | {config_gen['top_k']} |
| seed | {config_gen['seed']} |

## 2. Dataset

| Item | Value |
|---|---:|
| Training stories | {fmt(dataset.get('train_stories'), 0)} |
| Validation stories | {fmt(dataset.get('valid_stories'), 0)} |
| Training tokens | {fmt(dataset.get('train_tokens'), 0)} |
| Validation tokens | {fmt(dataset.get('valid_tokens'), 0)} |
| Vocabulary size | {dataset['vocab_size']} |
| Validation text source | {dataset.get('validation_source')} |

## 3. Validation Loss and Perplexity

Evaluated batches: {fmt(validation['batches_evaluated'], 0)}. Evaluated target tokens: {fmt(validation['tokens_evaluated'], 0)}.

| Metric | {version_label} |
|---|---:|
| Validation Loss | {fmt(validation['loss'])} |
| Validation Perplexity | {fmt(validation['perplexity'], 2)} |
| Parameters | {model['parameter_count']:,} |
| Block Size | {model['block_size']} tokenizer units |
| Vocab Size | {model['vocab_size']} |

Baseline comparison (perplexity is not story quality):

| Model | Validation Loss | Perplexity |
|---|---:|---:|
{baseline_table}

## 4. Fixed Story Prompt Suite

{chr(10).join(suite_sections)}

## 5. Repetition Metrics

Word-level n-gram rates on the continuation after the prompt.

| Prompt | Repeated 2-gram | Repeated 3-gram | Repeated 4-gram | Max phrase repeats |
|---|---:|---:|---:|---:|
{repetition_rows}

## 6. Diversity Metrics

| Prompt | Distinct-1 | Distinct-2 | Distinct-3 |
|---|---:|---:|---:|
{diversity_rows}

## 7. Multi-Seed Test

Prompt: `{multi['prompt']}`  
Seeds: {multi['seeds']}  
Settings: temperature={multi['temperature']}, top_k={multi['top_k']}, max_new_tokens={multi['max_new_tokens']}

| Metric | Result |
|---|---:|
| Unique outputs | {multi['unique_outputs']} / {multi['sample_count']} |
| Average Distinct-1 | {fmt(multi['average_distinct_1'], 3)} |
| Average Distinct-2 | {fmt(multi['average_distinct_2'], 3)} |
| Average repeated 3-gram rate | {fmt(multi['average_repeated_3gram_rate'], 3)} |

## 8. Sampling Sensitivity

Fixed prompt: `{SENSITIVITY_PROMPT}` · seed `{config_gen['seed']}`

| Temperature | Top-k | Distinct-2 | Repeated 3-gram Rate | Tokens/sec |
|---:|---:|---:|---:|---:|
{sensitivity_rows}

## 9. Generation Performance

| Metric | Result |
|---|---:|
| Mean suite tokens/sec | {fmt(performance['mean_suite_tokens_per_second'], 1)} |
| Mean suite characters/sec | {fmt(performance.get('mean_suite_characters_per_second'), 1)} |
| Peak inference GPU memory | {fmt(performance.get('peak_inference_gpu_memory_mb'), 2)} MB |
| Training tokens/sec (from history, if available) | {fmt(performance.get('training_tokens_per_second'), 0)} |
| Training characters/sec (from history, if available) | {fmt(performance.get('training_characters_per_second'), 0)} |
| Peak training GPU memory (from history, if available) | {fmt(performance.get('peak_training_gpu_memory_mb'), 2)} MB |

## 10. Manual Story Quality Rubric

Scores are left `null` until a human fills them. Scale: 0 = poor, 1 = partially acceptable, 2 = good.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Coherence | unrelated or broken continuation | partially connected | clearly continues the prompt |
| Grammar | mostly malformed | understandable with frequent errors | mostly grammatical |
| Story progression | no meaningful progression | some event development | clear continuation with events/actions |
| Repetition | severe repetition | noticeable repetition | little problematic repetition |

Each story-suite sample in `story_eval_v1.json` has:

```json
{{
  "coherence_score": null,
  "grammar_score": null,
  "story_progression_score": null,
  "repetition_score": null
}}
```

## 11. Baseline Summary

| Field | Value |
|---|---:|
| Parameter count | {fmt(baseline.get('parameter_count'), 0)} |
| Validation loss | {fmt(baseline.get('validation_loss'))} |
| Validation perplexity | {fmt(baseline.get('validation_perplexity'), 2)} |
| Context length | {fmt(baseline.get('context_length'), 0)} tokenizer units |
| Tokenizer type | {baseline.get('tokenizer_type')} |
| Generation speed (tokens/sec) | {fmt(baseline.get('generation_tokens_per_second'), 1)} |
| Generation speed (characters/sec) | {fmt(baseline.get('generation_characters_per_second'), 1)} |
| Training speed (tokens/sec) | {fmt(baseline.get('training_tokens_per_second'), 0)} |
| Training speed (characters/sec) | {fmt(baseline.get('training_characters_per_second'), 0)} |

## 12. Limitations

- Context length is {model['block_size']} tokenizer units (character tokens and BPE tokens are not comparable as text span)
- Tokens/sec alone is misleading across tokenizer types; prefer characters/sec
- TinyStories next-token perplexity is not a full story-quality score
- Manual rubric scores are not filled automatically
- Architecture compute benchmarks are separate files and were not replaced

Preserved computational benchmarks (do not delete):

```text
experiments/single_head_benchmark.*
experiments/single_head_128_benchmark.*
experiments/multi_head_benchmark.*
experiments/transformer_block_benchmark.*
experiments/minigpt_forward_benchmark.*
```

## 13. Metrics to Beat Next

- lower TinyStories validation loss/perplexity under controlled settings
- higher Distinct-2 / Distinct-3 without becoming incoherent
- lower repeated n-gram rate
- better manually rated coherence
- better manually rated grammar
- better story progression
- longer usable context
- better stopping behavior
- acceptable generation throughput in characters/sec

Do not assign numerical pass bars until human review.
"""


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_type = resolve_tokenizer_type()
    use_boundaries = tokenizer_type == "bpe"
    print("Device:", device.type)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("Tokenizer type:", tokenizer_type)
    print("Checkpoint:", config.BEST_CHECKPOINT)

    tokenizer = load_tokenizer()
    model, checkpoint = load_model(config.BEST_CHECKPOINT, device)
    model_config = checkpoint["model_config"]
    if model_config.get("tokenizer_type") and model_config["tokenizer_type"] != tokenizer_type:
        raise RuntimeError(
            f"Checkpoint tokenizer_type={model_config['tokenizer_type']!r} does not match "
            f"config.TOKENIZER_TYPE={tokenizer_type!r}."
        )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    history = load_json(config.HISTORY_PATH) or {}
    final_eval = load_json(config.FINAL_EVAL_JSON) or {}

    print("Loading validation stories...")
    valid_stories, valid_meta = load_validation_stories()
    valid_ids = encode_stories(tokenizer, valid_stories, use_story_boundaries=use_boundaries)
    valid_loader = DataLoader(
        TextDataset(valid_ids, model.block_size),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    print("Evaluating TinyStories validation loss...")
    val_loss, batches = average_validation_loss(
        model, valid_loader, device, config.EVAL_MAX_BATCHES
    )
    tokens_evaluated = batches * config.BATCH_SIZE * model.block_size
    val_ppl = perplexity(val_loss)

    if tokenizer_type == "char" and final_eval.get("baselines"):
        baselines = {
            "uniform_loss": final_eval["baselines"].get("uniform_loss"),
            "uniform_perplexity": final_eval["baselines"].get("uniform_perplexity"),
            "untrained_loss": final_eval["baselines"].get("untrained_loss"),
            "untrained_perplexity": final_eval["baselines"].get("untrained_perplexity"),
            "trained_loss": final_eval["baselines"].get("trained_loss"),
            "trained_perplexity": final_eval["baselines"].get("trained_perplexity"),
            "source": config.FINAL_EVAL_JSON,
        }
    else:
        baselines = {
            "uniform_loss": math.log(model.vocab_size),
            "uniform_perplexity": float(model.vocab_size),
            "untrained_loss": None,
            "untrained_perplexity": None,
            "trained_loss": val_loss,
            "trained_perplexity": val_ppl,
            "source": "computed_for_active_tokenizer",
        }

    gen_cfg = {
        "max_new_tokens": config.STORY_EVAL_MAX_NEW_TOKENS,
        "temperature": config.STORY_EVAL_TEMPERATURE,
        "top_k": config.STORY_EVAL_TOP_K,
        "seed": config.STORY_EVAL_SEED,
    }

    print("Running fixed story prompt suite...")
    story_suite = []
    for prompt in STORY_PROMPTS:
        sample = generate_once(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"],
            seed=gen_cfg["seed"],
            device=device,
        )
        annotated = annotate_sample(sample, tokenizer)
        story_suite.append(annotated)
        safe_print(f"[{prompt}] {sample['generated_text'][:160]}")

    print("Running multi-seed diversity test...")
    seeds = list(range(gen_cfg["seed"], gen_cfg["seed"] + MULTI_SEED_COUNT))
    multi_samples = []
    for seed in seeds:
        sample = generate_once(
            model,
            tokenizer,
            prompt=MULTI_SEED_PROMPT,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"],
            seed=seed,
            device=device,
        )
        multi_samples.append(annotate_sample(sample, tokenizer))

    multi_seed = {
        "prompt": MULTI_SEED_PROMPT,
        "seeds": seeds,
        "temperature": gen_cfg["temperature"],
        "top_k": gen_cfg["top_k"],
        "max_new_tokens": gen_cfg["max_new_tokens"],
        "sample_count": len(multi_samples),
        "unique_outputs": len({item["generated_text"] for item in multi_samples}),
        "average_distinct_1": mean_or_none(
            [item["diagnostics"]["distinct_1"] for item in multi_samples]
        ),
        "average_distinct_2": mean_or_none(
            [item["diagnostics"]["distinct_2"] for item in multi_samples]
        ),
        "average_repeated_3gram_rate": mean_or_none(
            [item["diagnostics"]["repeated_3gram_rate"] for item in multi_samples]
        ),
        "samples": multi_samples,
    }

    print("Running sampling sensitivity test...")
    sensitivity_settings = [
        {"temperature": 0.5, "top_k": 20},
        {"temperature": 0.8, "top_k": 20},
        {"temperature": 1.0, "top_k": 20},
        {"temperature": 1.2, "top_k": 20},
        {"temperature": 0.8, "top_k": 5},
        {"temperature": 0.8, "top_k": 40},
        {"temperature": 0.8, "top_k": None},
    ]
    sensitivity = []
    seen = set()
    for settings in sensitivity_settings:
        key = (settings["temperature"], settings["top_k"])
        if key in seen:
            continue
        seen.add(key)
        sample = generate_once(
            model,
            tokenizer,
            prompt=SENSITIVITY_PROMPT,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            seed=gen_cfg["seed"],
            device=device,
        )
        sensitivity.append(annotate_sample(sample, tokenizer))

    peak_infer = [
        item["peak_gpu_memory_mb"]
        for item in story_suite + multi_samples + sensitivity
        if item.get("peak_gpu_memory_mb") is not None
    ]
    mean_speed = mean_or_none([item["tokens_per_second"] for item in story_suite])
    mean_char_speed = mean_or_none(
        [item.get("characters_per_second") for item in story_suite]
    )

    result = {
        "model": {
            "tokenizer_type": tokenizer_type,
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "dropout": model_config["dropout"],
            "parameter_count": parameter_count,
            "checkpoint_path": config.BEST_CHECKPOINT,
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_train_loss": checkpoint.get("train_loss"),
            "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        },
        "dataset": {
            "train_stories": history.get("train_stories"),
            "valid_stories": valid_meta.get("valid_stories") or history.get("valid_stories"),
            "train_tokens": history.get("train_tokens"),
            "valid_tokens": len(valid_ids),
            "vocab_size": model.vocab_size,
            "validation_source": valid_meta.get("source"),
            "story_boundaries": use_boundaries,
        },
        "validation": {
            "loss": val_loss,
            "perplexity": val_ppl,
            "batches_evaluated": batches,
            "tokens_evaluated": tokens_evaluated,
        },
        "baselines": baselines,
        "generation_config": gen_cfg,
        "story_suite": story_suite,
        "multi_seed_diversity": multi_seed,
        "sampling_sensitivity": sensitivity,
        "performance": {
            "mean_suite_tokens_per_second": mean_speed,
            "mean_suite_characters_per_second": mean_char_speed,
            "peak_inference_gpu_memory_mb": max(peak_infer) if peak_infer else None,
            "training_tokens_per_second": history.get("tokens_per_second"),
            "training_characters_per_second": history.get("characters_per_second"),
            "peak_training_gpu_memory_mb": history.get("peak_gpu_memory_mb"),
        },
        "manual_rubric": empty_manual_rubric(),
        "v1_baseline_summary": {
            "parameter_count": parameter_count,
            "validation_loss": val_loss,
            "validation_perplexity": val_ppl,
            "context_length": model.block_size,
            "tokenizer_type": tokenizer_type,
            "generation_tokens_per_second": mean_speed,
            "generation_characters_per_second": mean_char_speed,
            "training_tokens_per_second": history.get("tokens_per_second"),
            "training_characters_per_second": history.get("characters_per_second"),
        },
        "preserved_compute_benchmarks": [
            "experiments/single_head_benchmark.json",
            "experiments/single_head_128_benchmark.json",
            "experiments/multi_head_benchmark.json",
            "experiments/transformer_block_benchmark.json",
            "experiments/minigpt_forward_benchmark.json",
        ],
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    json_path = Path(config.STORY_EVAL_JSON)
    report_path = Path(config.STORY_EVAL_REPORT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_markdown(result), encoding="utf-8")

    print()
    print("Validation loss:", f"{val_loss:.4f}")
    print("Validation perplexity:", f"{val_ppl:.2f}")
    print("Story suite prompts:", len(story_suite))
    print("Saved story eval JSON:")
    print(json_path.as_posix())
    print("Saved story eval report:")
    print(report_path.as_posix())


if __name__ == "__main__":
    main()
