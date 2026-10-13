"""Reusable TinyStories story-generation evaluation harness.

Works with character (v1) and BPE (v2, v3) profiles selected by config.EXPERIMENT.
Does not retrain the model. BPE runs also score a long-prefix memory probe and,
when both checkpoints exist, write the v2 vs v3 story scoreboard.
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
from generate import generate_from_ids, generate_once, load_model
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


# Sentence-initial words that look capitalized but are not character names.
_SENTENCE_STARTERS = {
    "a",
    "after",
    "all",
    "an",
    "and",
    "at",
    "but",
    "every",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "it",
    "little",
    "on",
    "once",
    "one",
    "she",
    "so",
    "some",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "to",
    "very",
    "when",
    "with",
}

# Closed-class words and very common story verbs. Remaining alphabetic words
# stand in for concrete nouns; this harness does not run a part-of-speech tagger.
_FUNCTION_WORDS = _SENTENCE_STARTERS | {
    "about",
    "again",
    "also",
    "are",
    "around",
    "as",
    "away",
    "back",
    "be",
    "because",
    "been",
    "before",
    "by",
    "came",
    "can",
    "could",
    "did",
    "do",
    "does",
    "down",
    "each",
    "else",
    "ever",
    "get",
    "got",
    "had",
    "has",
    "have",
    "here",
    "him",
    "how",
    "into",
    "is",
    "its",
    "just",
    "like",
    "made",
    "make",
    "many",
    "more",
    "much",
    "my",
    "not",
    "now",
    "of",
    "off",
    "only",
    "or",
    "our",
    "out",
    "over",
    "said",
    "than",
    "them",
    "these",
    "those",
    "through",
    "too",
    "under",
    "up",
    "us",
    "was",
    "we",
    "went",
    "were",
    "what",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "would",
    "you",
    "your",
}


def experiment_label(name: str | None, tokenizer_type: str) -> str:
    if name in {"v2_1_ctx256", "v3_ctx256"}:
        return "v2.1 context 256"
    if name == "v2_2_ctx256_longtrain":
        return "v2.2 context 256"
    if name == "v2_bpe" or tokenizer_type == "bpe":
        return "v2 BPE"
    return "v1"


def _iter_words(text: str):
    sentence_start = True
    for match in re.finditer(r"[A-Za-z]+|[.!?]", text):
        token = match.group()
        if token in ".!?":
            sentence_start = True
            continue
        yield token, sentence_start
        sentence_start = False


def extract_memory_terms(text: str) -> list[dict]:
    """Names, then other content words, from a decoded token span."""
    names = []
    contents = []
    seen: set[str] = set()
    for word, sentence_start in _iter_words(text):
        key = word.lower()
        if key in seen:
            continue
        is_name = (
            len(word) >= 3
            and word[0].isupper()
            and word[1:].islower()
            and not (sentence_start and key in _SENTENCE_STARTERS)
        )
        if is_name:
            seen.add(key)
            names.append({"term": word, "kind": "name"})
            continue
        if len(word) >= 3 and word.isalpha() and key not in _FUNCTION_WORDS:
            seen.add(key)
            contents.append({"term": word.lower(), "kind": "content"})
    return names + contents


def contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None


def exclusive_memory_terms(hidden_text: str, visible_text: str) -> list[dict]:
    """Keep terms from the opening that the recent context does not already repeat."""
    return [
        item
        for item in extract_memory_terms(hidden_text)
        if not contains_term(visible_text, item["term"])
    ]


def select_memory_items(stories: list[str], tokenizer) -> list[dict]:
    """Pick validation openings whose first 64 tokens are outside a 64-token model."""
    prefix_len = config.STORY_MEMORY_PREFIX_TOKENS
    hidden_len = config.STORY_MEMORY_HIDDEN_TOKENS
    if hidden_len < 1 or hidden_len >= prefix_len:
        raise ValueError("STORY_MEMORY_HIDDEN_TOKENS must be shorter than the prefix.")
    items = []
    for story_index, story in enumerate(stories):
        text = story.strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        if len(ids) < prefix_len:
            continue
        hidden_text = tokenizer.decode(ids[:hidden_len], skip_special_tokens=True)
        visible_text = tokenizer.decode(ids[hidden_len:prefix_len], skip_special_tokens=True)
        prefix_text = tokenizer.decode(ids[:prefix_len], skip_special_tokens=True)
        terms = exclusive_memory_terms(hidden_text, visible_text)
        if not terms:
            continue
        items.append(
            {
                "story_index": story_index,
                "prefix_ids": ids[:prefix_len],
                "prefix_text": prefix_text,
                "hidden_text": hidden_text,
                "terms": terms,
            }
        )
        if len(items) == config.STORY_MEMORY_STORY_COUNT:
            break
    if not items:
        raise RuntimeError(
            "No validation story had a 128-token prefix with memory terms "
            "that are absent from the recent 64 tokens."
        )
    return items


def run_story_suite(model, tokenizer, device: torch.device, gen_cfg: dict) -> list[dict]:
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
        story_suite.append(annotate_sample(sample, tokenizer))
        safe_print(f"[{prompt}] {sample['generated_text'][:160]}")
    return story_suite


def run_memory_probe(
    model,
    tokenizer,
    items: list[dict],
    gen_cfg: dict,
    device: torch.device,
    visible_tokens: int | None = None,
) -> dict:
    """Score the fixed 128-token probe.

    visible_tokens limits how much of that prefix the model receives, taken from
    the end. The tracked terms stay the terms from the original probe. They are
    not recomputed for the shorter view. A request above 128 tokens still sees
    only the 128-token prefix, because that is the whole probe prompt.
    """
    samples = []
    retained_total = 0
    term_total = 0
    visible_used = None
    for item in items:
        prefix_ids = list(item["prefix_ids"])
        prompt_text = item["prefix_text"]
        if visible_tokens is not None:
            visible_used = min(visible_tokens, len(prefix_ids))
            prefix_ids = prefix_ids[-visible_used:]
            prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        sample = generate_from_ids(
            model,
            tokenizer,
            prompt_ids=prefix_ids,
            prompt_text=prompt_text,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_k=gen_cfg["top_k"],
            seed=gen_cfg["seed"],
            device=device,
        )
        continuation = sample["continuation_text"]
        retained = [
            term["term"] for term in item["terms"] if contains_term(continuation, term["term"])
        ]
        retained_total += len(retained)
        term_total += len(item["terms"])
        annotated = annotate_sample(sample, tokenizer)
        annotated["story_index"] = item["story_index"]
        annotated["memory_terms"] = item["terms"]
        annotated["retained_terms"] = retained
        annotated["retention_rate"] = len(retained) / len(item["terms"])
        annotated["hidden_text"] = item["hidden_text"]
        samples.append(annotated)
        safe_print(
            f"[memory story {item['story_index']}] "
            f"retained {len(retained)}/{len(item['terms'])}"
        )
    eos_hits = sum(1 for sample in samples if sample["stopping"]["stopped_naturally"])
    return {
        "prefix_tokens": config.STORY_MEMORY_PREFIX_TOKENS,
        "hidden_tokens": config.STORY_MEMORY_HIDDEN_TOKENS,
        "visible_tokens_requested": visible_tokens,
        "visible_tokens_used": visible_used if visible_tokens is not None else config.STORY_MEMORY_PREFIX_TOKENS,
        "story_count": len(samples),
        "terms_total": term_total,
        "terms_retained": retained_total,
        "retention_rate": retained_total / term_total if term_total else None,
        "eos_stop_rate": eos_hits / len(samples) if samples else None,
        "samples": samples,
    }


def measure_validation(model, token_ids: list[int], block_size: int, device: torch.device) -> dict:
    loader = DataLoader(
        TextDataset(token_ids, block_size),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    loss, batches = average_validation_loss(model, loader, device, config.EVAL_MAX_BATCHES)
    return {
        "loss": loss,
        "perplexity": perplexity(loss),
        "block_size": block_size,
        "batches_evaluated": batches,
        "tokens_evaluated": batches * config.BATCH_SIZE * block_size,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def sample_continuation(item: dict) -> str:
    """Text generated after the prompt. Short-suite samples store the full string."""
    continuation = item.get("continuation_text")
    if continuation:
        return continuation
    prompt = item.get("prompt") or ""
    text = item.get("generated_text") or ""
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


def continuation_length_stats(suite: list[dict]) -> dict:
    """Length of the short-prompt suite, with EOS counted separately from the words before it."""
    lengths = [int(item["stopping"]["tokens_appended"]) for item in suite]
    before_eos = [
        int(item["stopping"]["tokens_appended"]) - 1
        for item in suite
        if item["stopping"]["stopped_naturally"]
    ]
    very_short = []
    for item in suite:
        if not item["stopping"]["stopped_naturally"]:
            continue
        tokens_before = int(item["stopping"]["tokens_appended"]) - 1
        continuation = sample_continuation(item)
        if tokens_before <= 5 or len(continuation.strip()) <= 2:
            very_short.append(
                {
                    "prompt": item.get("prompt", ""),
                    "continuation": continuation,
                    "tokens_before_eos": tokens_before,
                }
            )
    return {
        "mean_continuation_tokens": mean_or_none(lengths),
        "median_continuation_tokens": _median(lengths),
        "min_continuation_tokens": min(lengths) if lengths else None,
        "max_continuation_tokens": max(lengths) if lengths else None,
        "avg_tokens_before_eos": mean_or_none(before_eos),
        "median_tokens_before_eos": _median(before_eos),
        "eos_sample_count": len(before_eos),
        "extremely_short_eos_count": len(very_short),
        "extremely_short_eos_examples": very_short,
    }


def new_named_entities(prefix_text: str, continuation: str) -> list[str]:
    """Capitalized names in the continuation that do not appear in the prefix.

    This is an exact-name check. It does not decide that a common noun is the
    same character as a name.
    """
    found = []
    seen = set()
    for term in extract_memory_terms(continuation):
        if term["kind"] != "name" or term["term"].lower() in _NOT_CHARACTER_NAMES:
            continue
        key = term["term"].lower()
        if key in seen or contains_term(prefix_text, term["term"]):
            continue
        # "Don't" is split into "Don" by the letter scanner. That is not a new character.
        if re.search(rf"\b{re.escape(term['term'])}['’]", continuation):
            continue
        seen.add(key)
        found.append(term["term"])
    return found


def unrelated_name_stats(samples: list[dict]) -> dict:
    """Share of long-prefix stories whose continuation introduces a new name."""
    records = []
    for sample in samples:
        prefix = sample.get("prompt") or ""
        continuation = sample.get("continuation_text") or ""
        names = new_named_entities(prefix, continuation)
        records.append(
            {
                "story_index": sample.get("story_index"),
                "new_named_entities": names,
                "introduced_unrelated_name": bool(names),
            }
        )
    count = len(records)
    introduced = sum(1 for record in records if record["introduced_unrelated_name"])
    return {
        "unrelated_name_introduction_rate": introduced / count if count else None,
        "stories_with_new_name": introduced,
        "story_count": count,
        "records": records,
    }


def suite_metrics(suite: list[dict]) -> dict:
    eos_flags = [bool(item["stopping"]["stopped_naturally"]) for item in suite]
    peaks = [
        item["peak_gpu_memory_mb"]
        for item in suite
        if item.get("peak_gpu_memory_mb") is not None
    ]
    lengths = continuation_length_stats(suite)
    return {
        "distinct_2": mean_or_none([item["diagnostics"]["distinct_2"] for item in suite]),
        "distinct_3": mean_or_none([item["diagnostics"]["distinct_3"] for item in suite]),
        "repeated_3gram_rate": mean_or_none(
            [item["diagnostics"]["repeated_3gram_rate"] for item in suite]
        ),
        "eos_stop_rate": sum(eos_flags) / len(eos_flags) if eos_flags else None,
        "characters_per_second": mean_or_none(
            [item.get("characters_per_second") for item in suite]
        ),
        "peak_gpu_memory_mb": max(peaks) if peaks else None,
        "prompt_count": len(suite),
        **lengths,
    }


def scoreboard_partner() -> tuple[str, str] | None:
    """The other BPE checkpoint to place beside the active run. Does not train it."""
    if config.EXPERIMENT == "v2_1_ctx256":
        return config.V2_BPE_BEST_CHECKPOINT, "v2_bpe"
    if config.EXPERIMENT == "v2_bpe":
        return config.V2_1_CTX256_BEST_CHECKPOINT, "v2_1_ctx256"
    return None


def scoreboard_side(
    experiment: str,
    checkpoint_path: str,
    model,
    validation: dict,
    validation_context_64: dict,
    suite: list[dict],
    probe: dict,
) -> dict:
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = suite_metrics(suite)
    name_stats = unrelated_name_stats(probe["samples"])
    return {
        "experiment": experiment,
        "label": experiment_label(experiment, "bpe"),
        "checkpoint": checkpoint_path,
        "block_size": model.block_size,
        "position_encoding": getattr(model, "position_encoding", "absolute"),
        "parameter_count": parameter_count,
        "validation_loss": validation["loss"],
        "validation_perplexity": validation["perplexity"],
        "validation_block_size": validation["block_size"],
        "validation_loss_context_64": validation_context_64["loss"],
        "validation_perplexity_context_64": validation_context_64["perplexity"],
        "long_prefix_retention_rate": probe["retention_rate"],
        "long_prefix_terms_retained": probe["terms_retained"],
        "long_prefix_terms_total": probe["terms_total"],
        "long_prefix_story_count": probe["story_count"],
        "long_prefix_eos_stop_rate": probe["eos_stop_rate"],
        "short_suite_distinct_2": metrics["distinct_2"],
        "short_suite_distinct_3": metrics["distinct_3"],
        "short_suite_repeated_3gram_rate": metrics["repeated_3gram_rate"],
        "eos_stop_rate": metrics["eos_stop_rate"],
        "avg_tokens_before_eos": metrics["avg_tokens_before_eos"],
        "median_tokens_before_eos": metrics["median_tokens_before_eos"],
        "mean_continuation_tokens": metrics["mean_continuation_tokens"],
        "median_continuation_tokens": metrics["median_continuation_tokens"],
        "min_continuation_tokens": metrics["min_continuation_tokens"],
        "max_continuation_tokens": metrics["max_continuation_tokens"],
        "extremely_short_eos_count": metrics["extremely_short_eos_count"],
        "extremely_short_eos_examples": metrics["extremely_short_eos_examples"],
        "unrelated_name_introduction_rate": name_stats["unrelated_name_introduction_rate"],
        "unrelated_name_records": name_stats["records"],
        "characters_per_second": metrics["characters_per_second"],
        "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"],
        "memory_stories": [
            {
                "story_index": sample["story_index"],
                "terms": [term["term"] for term in sample["memory_terms"]],
                "retained_terms": sample["retained_terms"],
                "retention_rate": sample["retention_rate"],
                "stopped_on_eos": sample["stopping"]["stopped_naturally"],
                "continuation_text": sample.get("continuation_text", ""),
                "prompt_text": sample.get("prompt", ""),
                "tokens_appended": sample["stopping"]["tokens_appended"],
                "new_named_entities": new_named_entities(
                    sample.get("prompt") or "",
                    sample.get("continuation_text") or "",
                ),
            }
            for sample in probe["samples"]
        ],
    }


def build_scoreboard(v2: dict, v3: dict) -> str:
    """One row per metric. Manual rubric scores stay unset."""

    def row(label: str, key: str, digits: int = 4) -> str:
        return f"| {label} | {fmt(v2.get(key), digits)} | {fmt(v3.get(key), digits)} |"

    retention_line = (
        f"Long-prefix retention is {fmt(v2['long_prefix_retention_rate'], 3)} for v2 "
        f"({fmt(v2['long_prefix_terms_retained'], 0)}/{fmt(v2['long_prefix_terms_total'], 0)} terms) "
        f"and {fmt(v3['long_prefix_retention_rate'], 3)} for v3 "
        f"({fmt(v3['long_prefix_terms_retained'], 0)}/{fmt(v3['long_prefix_terms_total'], 0)} terms)."
    )
    loss_line = (
        f"Validation loss at each model's training context is {fmt(v2['validation_loss'])} "
        f"for v2 (context {fmt(v2['validation_block_size'], 0)}) and {fmt(v3['validation_loss'])} "
        f"for v3 (context {fmt(v3['validation_block_size'], 0)}). "
        f"On matched 64-token windows the losses are {fmt(v2['validation_loss_context_64'])} "
        f"and {fmt(v3['validation_loss_context_64'])}."
    )
    repetition_line = (
        f"Short-suite repeated 3-gram rate is {fmt(v2['short_suite_repeated_3gram_rate'], 3)} "
        f"for v2 and {fmt(v3['short_suite_repeated_3gram_rate'], 3)} for v3. "
        f"Short-suite Distinct-2 is {fmt(v2['short_suite_distinct_2'], 3)} "
        f"and {fmt(v3['short_suite_distinct_2'], 3)}."
    )
    return f"""# Story scoreboard: v2 BPE vs v3 context 256

Same BPE tokenizer, same 5,000/500 TinyStories slice, same 1,000 AdamW steps at 3e-4, and the same 2,048 tokens per step. v2 uses batch 32 and context 64. v3 uses batch 8 and context 256. Decoding for every row is temperature {config.STORY_EVAL_TEMPERATURE}, top-k {config.STORY_EVAL_TOP_K}, seed {config.STORY_EVAL_SEED}.

The long-prefix probe prompts with the first {config.STORY_MEMORY_PREFIX_TOKENS} BPE tokens of a validation story. v2 generation keeps only the last {config.STORY_MEMORY_HIDDEN_TOKENS} of those tokens. Scored terms come from that hidden opening and do not already appear in the recent {config.STORY_MEMORY_HIDDEN_TOKENS} tokens.

| Metric | v2 | v3 |
|---|---:|---:|
{row("Validation loss (training context)", "validation_loss")}
{row("Validation perplexity (training context)", "validation_perplexity", 2)}
{row("Validation loss (64-token windows)", "validation_loss_context_64")}
{row("Validation perplexity (64-token windows)", "validation_perplexity_context_64", 2)}
{row("Long-prefix retention", "long_prefix_retention_rate", 3)}
{row("Short-suite Distinct-2", "short_suite_distinct_2", 3)}
{row("Short-suite repeated 3-gram rate", "short_suite_repeated_3gram_rate", 3)}
{row("Short-suite EOS stop rate", "eos_stop_rate", 3)}
{row("Short-suite characters/sec", "characters_per_second", 1)}
{row("Peak inference GPU memory (MB)", "peak_gpu_memory_mb", 1)}
| Parameters | {fmt(v2['parameter_count'], 0)} | {fmt(v3['parameter_count'], 0)} |
| Context length | {fmt(v2['block_size'], 0)} | {fmt(v3['block_size'], 0)} |

## Reading

{retention_line}

{loss_line}

{repetition_line}

Manual coherence, grammar, story-progression, and repetition scores stay null until a person fills them. A stronger story model has higher long-prefix retention, a 64-token validation loss that holds up, and a repeated 3-gram rate that does not jump.

Checkpoints: `{v2['checkpoint']}` and `{v3['checkpoint']}`.
"""


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

    version_label = experiment_label(model.get("experiment"), model.get("tokenizer_type", ""))
    probe = result.get("long_prefix_probe")
    if probe:
        probe_rows = "\n".join(
            f"| {item['story_index']} | "
            f"{', '.join(term['term'] for term in item['memory_terms'])} | "
            f"{', '.join(item['retained_terms']) or 'none'} | "
            f"{fmt(item['retention_rate'], 3)} | "
            f"{item['stopping']['stopped_naturally']} |"
            for item in probe["samples"]
        )
        probe_section = f"""## 4b. Long-prefix memory probe

Prompt length: {probe['prefix_tokens']} BPE tokens. Terms are character names and content words from the first {probe['hidden_tokens']} tokens that do not also appear in the recent {probe['hidden_tokens']} tokens. v2 can only attend to that recent span.

| Story index | Hidden-span terms | Retained in continuation | Retention | Stopped on EOS |
|---|---|---|---:|---|
{probe_rows}

Overall retention: {fmt(probe['retention_rate'], 3)} ({fmt(probe['terms_retained'], 0)}/{fmt(probe['terms_total'], 0)} terms). Probe EOS stop rate: {fmt(probe['eos_stop_rate'], 3)}.

"""
    else:
        probe_section = ""
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

{probe_section}## 5. Repetition Metrics

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

Each story-suite sample in the saved story-eval JSON has:

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


# Capitalized words that the fact check should not treat as character names.
_NOT_CHARACTER_NAMES = {
    "can",
    "finally",
    "first",
    "hey",
    "hi",
    "how",
    "let",
    "no",
    "now",
    "oh",
    "once",
    "please",
    "sure",
    "thank",
    "then",
    "upon",
    "what",
    "when",
    "where",
    "who",
    "why",
    "yes",
    "you",
    "look",
}


_GENERIC_PHRASE_WORDS = {
    "big",
    "day",
    "girl",
    "good",
    "happy",
    "little",
    "lived",
    "looked",
    "named",
    "one",
    "said",
    "time",
    "very",
    "wanted",
    "went",
}


def notable_phrases(text: str, limit: int = 4) -> list[str]:
    """Two-word spans from the prefix, kept as exact strings.

    "icy hill" counts only when those two words appear together. A later
    paraphrase such as "the cold slope" is not treated as the same phrase.
    """
    tokens = [word for word, _start in _iter_words(text)]
    found = []
    seen = set()
    for left, right in zip(tokens, tokens[1:]):
        if not (left.isalpha() and right.isalpha()):
            continue
        if len(left) < 3 or len(right) < 3:
            continue
        if left.lower() in _FUNCTION_WORDS or right.lower() in _FUNCTION_WORDS:
            continue
        if left.lower() in _GENERIC_PHRASE_WORDS or right.lower() in _GENERIC_PHRASE_WORDS:
            continue
        if left[:1].isupper() or right[:1].isupper():
            continue
        if left.lower() == right.lower():
            continue
        phrase = f"{left} {right}"
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(phrase)
    repeated = [phrase for phrase in found if len(re.findall(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE)) > 1]
    ordered = repeated + [phrase for phrase in found if phrase not in repeated]
    return ordered[:limit]


def continuation_fact_checks(prefix_text: str, continuation: str, early_terms: list[dict], stopped_on_eos: bool) -> dict:
    """Lexical checks. A missed paraphrase is not evidence that the story forgot the scene."""
    tracked = [term for term in early_terms if term["term"].lower() not in _NOT_CHARACTER_NAMES]
    mentions = [
        {
            "term": term["term"],
            "kind": term["kind"],
            "mentioned": contains_term(continuation, term["term"]),
        }
        for term in tracked
    ]
    prefix_names = [
        term["term"]
        for term in extract_memory_terms(prefix_text)
        if term["kind"] == "name" and term["term"].lower() not in _NOT_CHARACTER_NAMES
    ]
    phrases = notable_phrases(prefix_text)
    expected = prefix_names + [phrase for phrase in phrases if phrase not in prefix_names]
    exact_mentions = []
    for entity in expected:
        if " " in entity:
            if re.search(rf"\b{re.escape(entity)}\b", continuation, flags=re.IGNORECASE):
                exact_mentions.append(entity)
        elif contains_term(continuation, entity):
            exact_mentions.append(entity)
    new_names = new_named_entities(prefix_text, continuation)
    prefix_names_mentioned = [name for name in prefix_names if contains_term(continuation, name)]
    return {
        "mentions": mentions,
        "new_names": new_names,
        "unrelated_new_name": bool(new_names),
        "expected_entities": expected,
        "exact_mentions": exact_mentions,
        "new_named_entities": new_names,
        "unrelated_new_protagonist": bool(new_names) and not prefix_names_mentioned,
        "eos": stopped_on_eos,
        "stopped_on_eos": stopped_on_eos,
    }


def collect_scoreboard_model(
    experiment: str,
    checkpoint_path: str,
    model,
    valid_ids: list[int],
    memory_items: list[dict],
    gen_cfg: dict,
    device: torch.device,
    tokenizer,
    suite: list[dict] | None = None,
    probe: dict | None = None,
    native: dict | None = None,
    matched_64: dict | None = None,
) -> dict:
    if native is None:
        native = measure_validation(model, valid_ids, model.block_size, device)
    if matched_64 is None:
        if model.block_size == config.STORY_MEMORY_HIDDEN_TOKENS:
            matched_64 = native
        else:
            matched_64 = measure_validation(
                model,
                valid_ids,
                config.STORY_MEMORY_HIDDEN_TOKENS,
                device,
            )
    if suite is None:
        print(f"Running {experiment} short story suite...")
        suite = run_story_suite(model, tokenizer, device, gen_cfg)
    if probe is None:
        print(f"Running {experiment} long-prefix probe...")
        probe = run_memory_probe(model, tokenizer, memory_items, gen_cfg, device)
    side = scoreboard_side(
        experiment,
        checkpoint_path,
        model,
        native,
        matched_64,
        suite,
        probe,
    )
    side["native_perplexity"] = native["perplexity"]
    side["shared_64_perplexity"] = matched_64["perplexity"]
    return side


def compare_real_prefixes(models: list[tuple[str, object]], tokenizer, stories: list[str], device: torch.device) -> list[dict]:
    """Same 256-token validation prefix for every model. v2 still crops to 64 inside generate()."""
    from evaluate_context_utilization import encode_validation_stories, select_story_examples

    encoded = encode_validation_stories(stories, tokenizer)
    examples = select_story_examples(encoded, tokenizer, config.CONTEXT_UTIL_EXAMPLE_COUNT, 256)
    compared = []
    for example in examples:
        prefix_ids = example["prefix_ids"]
        hidden_text = tokenizer.decode(prefix_ids[:192], skip_special_tokens=True)
        early_terms = [
            term
            for term in extract_memory_terms(hidden_text)
            if term["term"].lower() not in _NOT_CHARACTER_NAMES
        ]
        names = [term for term in early_terms if term["kind"] == "name"]
        contents = [term for term in early_terms if term["kind"] == "content"]
        early_terms = (names + contents)[:8]
        continuations = []
        for label, model in models:
            sample = generate_from_ids(
                model,
                tokenizer,
                prompt_ids=prefix_ids,
                prompt_text=example["prefix_text"],
                max_new_tokens=config.CONTEXT_UTIL_EXAMPLE_NEW_TOKENS,
                temperature=config.STORY_EVAL_TEMPERATURE,
                top_k=config.STORY_EVAL_TOP_K,
                seed=config.STORY_EVAL_SEED,
                device=device,
            )
            checks = continuation_fact_checks(
                example["prefix_text"],
                sample["continuation_text"],
                early_terms,
                sample["stopped_on_eos"],
            )
            tokens_generated = int(sample["tokens_generated"])
            tokens_before_eos = tokens_generated - 1 if sample["stopped_on_eos"] else None
            extremely_short = bool(
                sample["stopped_on_eos"]
                and (
                    (tokens_before_eos is not None and tokens_before_eos <= 5)
                    or len(sample["continuation_text"].strip()) <= 2
                )
            )
            continuations.append(
                {
                    "label": label,
                    "continuation_text": sample["continuation_text"],
                    "tokens_generated": tokens_generated,
                    "tokens_before_eos": tokens_before_eos,
                    "extremely_short": extremely_short,
                    "checks": checks,
                    "visible_limit": model.block_size,
                }
            )
        compared.append(
            {
                "story_index": example["story_index"],
                "prefix_text": example["prefix_text"],
                "early_terms": [term["term"] for term in early_terms],
                "continuations": continuations,
            }
        )
    return compared


def family_success(v2: dict, v21: dict, v22: dict) -> dict:
    loss_drop = v21["validation_loss_context_64"] - v22["validation_loss_context_64"]
    repetition_increase = (
        v22["short_suite_repeated_3gram_rate"] - v21["short_suite_repeated_3gram_rate"]
    )
    checks = {
        "retention_at_least_v2_1": v22["long_prefix_retention_rate"] >= v21["long_prefix_retention_rate"],
        "shared_64_loss_drop": loss_drop,
        "shared_64_loss_improves_clearly": loss_drop >= 0.05,
        "repeated_3gram_increase": repetition_increase,
        "repetition_not_materially_worse": repetition_increase <= 0.01,
        "eos_rate_above_v2": v22["eos_stop_rate"] - v2["eos_stop_rate"],
        "eos_substantially_above_v2": v22["eos_stop_rate"] >= v2["eos_stop_rate"] + 0.25,
    }
    checks["successful"] = bool(
        checks["retention_at_least_v2_1"]
        and checks["shared_64_loss_improves_clearly"]
        and checks["repetition_not_materially_worse"]
        and checks["eos_substantially_above_v2"]
    )
    return checks


def build_family_scoreboard(v2: dict, v21: dict, v22: dict, stories: list[dict], success: dict) -> str:
    def row(label: str, key: str, digits: int = 4) -> str:
        return (
            f"| {label} | {fmt(v2.get(key), digits)} | {fmt(v21.get(key), digits)} | "
            f"{fmt(v22.get(key), digits)} |"
        )

    story_blocks = []
    for story in stories:
        lines = [
            f"### Validation story {story['story_index']}",
            "",
            "Prefix:",
            "",
            "```text",
            story["prefix_text"],
            "```",
            "",
            "Early terms tracked from the opening, before the last 64 prefix tokens: "
            + (", ".join(story["early_terms"]) or "none"),
            "",
        ]
        for item in story["continuations"]:
            checks = item["checks"]
            mention_lines = "\n".join(
                f"- mentions {mention['term']}: {'yes' if mention['mentioned'] else 'no'}"
                for mention in checks["mentions"]
            )
            lines.extend(
                [
                    f"#### {item['label']} continuation",
                    "",
                    f"This model's context limit is {item['visible_limit']} tokens. "
                    "A shorter model keeps only the end of the prefix.",
                    "",
                    "```text",
                    item["continuation_text"],
                    "```",
                    "",
                    mention_lines or "- no tracked terms",
                    f"- unrelated new name: {'yes' if checks['unrelated_new_name'] else 'no'}"
                    + (f" ({', '.join(checks['new_names'])})" if checks["new_names"] else ""),
                    f"- produced EOS: {'yes' if checks['stopped_on_eos'] else 'no'}",
                    "",
                ]
            )
        story_blocks.append("\n".join(lines))

    if success["successful"]:
        verdict = (
            "v2.2 meets the stated checks: long-prefix retention stays at least as high as v2.1, "
            "shared 64-token loss drops by at least 0.05, repeated 3-gram rate does not rise by "
            "more than 0.01, and the EOS stop rate stays at least 0.25 above v2."
        )
    else:
        verdict = (
            "v2.2 does not meet every stated check. "
            f"Retention at least v2.1: {success['retention_at_least_v2_1']}. "
            f"Shared 64-token loss drop: {success['shared_64_loss_drop']:.4f} "
            f"(clear means at least 0.05). "
            f"Repeated 3-gram increase: {success['repeated_3gram_increase']:.4f} "
            f"(material means more than 0.01). "
            f"EOS rate above v2: {success['eos_rate_above_v2']:.3f} "
            f"(substantial means at least 0.25)."
        )
    return f"""# Story scoreboard: v2 vs v2.1 vs v2.2

v2 is the BPE model with context 64. v2.1 is the same width and data with context 256, trained for 1000 steps. v2.2 resumes v2.1 and continues to a 3000-step budget. Tokenizer, width, depth, learning rate, and decoding stay fixed. v2.2 changes total training tokens on purpose: 2048 tokens/step times 3000 steps is 6,144,000 target tokens.

Shared 64-token loss asks how well each model predicts when the window is only 64 tokens. Native-context loss asks how well each model predicts in the window it was trained for. Those numbers stay separate.

| Metric | v2 | v2.1 | v2.2 |
|---|---:|---:|---:|
{row("Shared 64-token validation loss", "validation_loss_context_64")}
{row("Native-context validation loss", "validation_loss")}
{row("Shared 64-token perplexity", "shared_64_perplexity", 2)}
{row("Native-context perplexity", "native_perplexity", 2)}
{row("Long-prefix retention", "long_prefix_retention_rate", 3)}
{row("Short-suite Distinct-2", "short_suite_distinct_2", 3)}
{row("Short-suite repeated 3-gram rate", "short_suite_repeated_3gram_rate", 3)}
{row("EOS stop rate", "eos_stop_rate", 3)}
{row("Characters/sec", "characters_per_second", 1)}
{row("Peak GPU memory (MB)", "peak_gpu_memory_mb", 1)}
| Context size | 64 | 256 | 256 |
| Batch size | 32 | 8 | 8 |
| Total training steps | 1,000 | 1,000 | 3,000 |
| Best checkpoint step | {fmt(v2.get("checkpoint_step"), 0)} | {fmt(v21.get("checkpoint_step"), 0)} | {fmt(v22.get("checkpoint_step"), 0)} |
| Target tokens / step | 2,048 | 2,048 | 2,048 |
| Total training target tokens | 2,048,000 | 2,048,000 | 6,144,000 |

## Reading the metrics on these continuations

Long-prefix retention counts early entity and content words that reappear as the same words. It misses a paraphrase: a prefix about Toby and a yellow ball can continue as "the dog dropped his toy" and still score as a miss.

Repeated 3-gram rate counts how often a three-word sequence is reused in the short-prompt suite. A high rate looks like the same sentence starting again. EOS stop rate is the share of those short prompts that end on `<eos>` instead of running into `max_new_tokens`.

The continuations below are the actual generations. Manual coherence scores are not assigned.

{chr(10).join(story_blocks)}

## v2.2 checks

{verdict}
"""


def write_family_scoreboard(
    active_model,
    active_checkpoint_step: int | None,
    tokenizer,
    valid_ids: list[int],
    valid_stories: list[str],
    memory_items: list[dict],
    gen_cfg: dict,
    device: torch.device,
    suite: list[dict],
    probe: dict,
    native: dict,
    matched_64: dict,
) -> None:
    v22 = collect_scoreboard_model(
        "v2_2_ctx256_longtrain",
        config.V2_2_CTX256_BEST_CHECKPOINT,
        active_model,
        valid_ids,
        memory_items,
        gen_cfg,
        device,
        tokenizer,
        suite=suite,
        probe=probe,
        native=native,
        matched_64=matched_64,
    )
    v22["checkpoint_step"] = active_checkpoint_step
    loaded = [("v2", config.V2_BPE_BEST_CHECKPOINT, "v2_bpe")]
    loaded.append(("v2.1", config.V2_1_CTX256_BEST_CHECKPOINT, "v2_1_ctx256"))
    sides = {"v2.2": v22}
    models = [("v2.2", active_model)]
    for label, path, experiment in loaded:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {path}. The family scoreboard does not train models.")
        print(f"Scoring {label} from {path}")
        model, checkpoint = load_model(path, device)
        side = collect_scoreboard_model(
            experiment,
            path,
            model,
            valid_ids,
            memory_items,
            gen_cfg,
            device,
            tokenizer,
        )
        side["checkpoint_step"] = checkpoint.get("step")
        sides[label] = side
        models.append((label, model))
    # Generate in version order: v2, v2.1, v2.2.
    ordered_models = [item for item in models if item[0] == "v2"]
    ordered_models += [item for item in models if item[0] == "v2.1"]
    ordered_models += [item for item in models if item[0] == "v2.2"]
    print("Generating shared validation-prefix comparisons...")
    stories = compare_real_prefixes(ordered_models, tokenizer, valid_stories, device)
    success = family_success(sides["v2"], sides["v2.1"], sides["v2.2"])
    payload = {"v2": sides["v2"], "v2_1": sides["v2.1"], "v2_2": sides["v2.2"], "stories": stories, "success": success}
    json_path = Path(config.STORY_SCOREBOARD_V2_V21_V22_JSON)
    report_path = Path(config.STORY_SCOREBOARD_V2_V21_V22_REPORT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        build_family_scoreboard(sides["v2"], sides["v2.1"], sides["v2.2"], stories, success),
        encoding="utf-8",
    )
    print("Saved family scoreboard:")
    print(report_path.as_posix())


def v23_success(v2: dict, v21: dict, v22: dict, v23: dict) -> dict:
    """v2.3 succeeds by using distant context better while staying near v2.2's loss.

    A lower validation loss by itself is not the goal. A higher EOS rate is not
    treated as an automatic improvement.
    """
    loss_gap = v23["validation_loss_context_64"] - v22["validation_loss_context_64"]
    checks = {
        "retention_above_v2_2": v23["long_prefix_retention_rate"] > v22["long_prefix_retention_rate"],
        "retention_at_least_v2_1": v23["long_prefix_retention_rate"] >= v21["long_prefix_retention_rate"],
        "shared_64_loss_gap_vs_v2_2": loss_gap,
        "shared_64_loss_near_v2_2": loss_gap <= 0.15,
        "unrelated_name_rate_below_v2_2": (
            v23["unrelated_name_introduction_rate"] < v22["unrelated_name_introduction_rate"]
        ),
    }
    checks["successful"] = bool(
        checks["retention_above_v2_2"]
        and checks["shared_64_loss_near_v2_2"]
        and checks["unrelated_name_rate_below_v2_2"]
    )
    checks["strong"] = bool(checks["successful"] and checks["retention_at_least_v2_1"])
    return checks


def build_v23_scoreboard(
    v2: dict,
    v21: dict,
    v22: dict,
    v23: dict,
    stories: list[dict],
    success: dict,
) -> str:
    sides = (v2, v21, v22, v23)

    def row(label: str, key: str, digits: int = 4) -> str:
        cells = " | ".join(fmt(side.get(key), digits) for side in sides)
        return f"| {label} | {cells} |"

    story_blocks = []
    for story in stories:
        lines = [
            f"### Validation story {story['story_index']}",
            "",
            "Prefix:",
            "",
            "```text",
            story["prefix_text"],
            "```",
            "",
        ]
        for item in story["continuations"]:
            checks = item["checks"]
            mention_lines = "\n".join(
                f"- exact mention {entity}: {'yes' if entity in checks['exact_mentions'] else 'no'}"
                for entity in checks.get("expected_entities", [])
            )
            short_note = ""
            if item.get("extremely_short"):
                short_note = (
                    f"- extremely short continuation: yes "
                    f"({item.get('tokens_before_eos')} tokens before EOS)\n"
                )
            lines.extend(
                [
                    f"#### {item['label']}",
                    "",
                    "```text",
                    item["continuation_text"],
                    "```",
                    "",
                    f"- expected entities: {', '.join(checks.get('expected_entities') or []) or 'none'}",
                    mention_lines or "- no expected entities",
                    f"- new named entities: {', '.join(checks.get('new_named_entities') or []) or 'none'}",
                    "- unrelated new protagonist: "
                    + ("yes" if checks.get("unrelated_new_protagonist") else "no"),
                    f"- produced EOS: {'yes' if checks.get('eos') else 'no'}",
                    short_note.rstrip(),
                    "",
                ]
            )
        story_blocks.append("\n".join(lines))

    short_lines = []
    for label, side in (("v2", v2), ("v2.1", v21), ("v2.2", v22), ("v2.3", v23)):
        examples = side.get("extremely_short_eos_examples") or []
        shortest = side.get("shortest_eos_example")
        if examples:
            shown = examples[0]
            short_lines.append(
                f"- {label}: {side.get('extremely_short_eos_count')} extremely short EOS "
                f"continuation(s). One of them follows `{shown['prompt']}` with "
                f"`{shown['continuation']}` ({shown['tokens_before_eos']} tokens before EOS)."
            )
        elif shortest:
            short_lines.append(
                f"- {label}: no continuation stops within 5 tokens. The shortest EOS "
                f"continuation follows `{shortest['prompt']}` with "
                f"`{shortest['continuation']}` ({shortest['tokens_before_eos']} tokens before EOS)."
            )
        else:
            short_lines.append(f"- {label}: no EOS continuation in the short suite.")

    if success["strong"]:
        verdict = (
            "v2.3 meets the strong result: long-prefix retention is above v2.2 and at least "
            "v2.1, shared 64-token loss stays within 0.15 of v2.2, and the unrelated-name "
            "introduction rate is lower than v2.2."
        )
    elif success["successful"]:
        verdict = (
            "v2.3 meets the main checks: retention is above v2.2, shared 64-token loss stays "
            "within 0.15 of v2.2, and fewer long-prefix stories introduce an unrelated name. "
            "Retention is still below v2.1, so this is not the strong result."
        )
    else:
        verdict = (
            "v2.3 does not meet the main checks. "
            f"Retention above v2.2: {success['retention_above_v2_2']}. "
            f"Retention at least v2.1: {success['retention_at_least_v2_1']}. "
            f"Shared 64-token loss gap versus v2.2: {success['shared_64_loss_gap_vs_v2_2']:.4f} "
            "(near means no more than 0.15 higher). "
            f"Unrelated-name rate below v2.2: {success['unrelated_name_rate_below_v2_2']}."
        )
    return f"""# Story scoreboard: v2 vs v2.1 vs v2.2 vs v2.3

v2, v2.1, and v2.2 use learned absolute positions. v2.3 uses the same width, depth, data, and 3000-step budget, with RoPE applied to Q and K and no learned position table. v2.3 is trained from fresh weights.

Shared 64-token loss and native-context loss stay in separate rows. Perplexity is exp(loss) on the same split. Long-prefix retention uses the existing 128-token probe and the same early terms. A higher EOS stop rate is listed, and the token counts show whether those stops are short.

| Metric | v2 | v2.1 | v2.2 | v2.3 |
|---|---:|---:|---:|---:|
{row("Shared 64-token validation loss", "validation_loss_context_64")}
{row("Native-context validation loss", "validation_loss")}
{row("Shared perplexity", "shared_64_perplexity", 2)}
{row("Native perplexity", "native_perplexity", 2)}
{row("Long-prefix retention", "long_prefix_retention_rate", 3)}
{row("Distinct-2", "short_suite_distinct_2", 3)}
{row("Distinct-3", "short_suite_distinct_3", 3)}
{row("Repeated 3-gram rate", "short_suite_repeated_3gram_rate", 3)}
{row("EOS stop rate", "eos_stop_rate", 3)}
{row("Avg tokens before EOS", "avg_tokens_before_eos", 1)}
{row("Median tokens before EOS", "median_tokens_before_eos", 1)}
{row("Mean continuation tokens", "mean_continuation_tokens", 1)}
{row("Median continuation tokens", "median_continuation_tokens", 1)}
{row("Min continuation tokens", "min_continuation_tokens", 0)}
{row("Max continuation tokens", "max_continuation_tokens", 0)}
{row("Unrelated-name introduction rate", "unrelated_name_introduction_rate", 3)}
{row("Characters/sec", "characters_per_second", 1)}
{row("Peak GPU memory (MB)", "peak_gpu_memory_mb", 1)}
| Position encoding | {v2.get("position_encoding", "absolute")} | {v21.get("position_encoding", "absolute")} | {v22.get("position_encoding", "absolute")} | {v23.get("position_encoding", "rope")} |

## Short continuations

An EOS stop with only a few tokens still counts in the EOS rate. These lines quote the short suite when that happens.

{chr(10).join(short_lines)}

## Real validation prefixes

Expected entities are names and two-word spans copied from the prefix. A mention counts only when that same string appears in the continuation. A new capitalized name that was not in the prefix is recorded as a new named entity. The check does not decide that a common noun is the same character.

{chr(10).join(story_blocks)}

## v2.3 checks

{verdict}
"""


def write_v23_scoreboard(
    active_model,
    active_checkpoint_step: int | None,
    tokenizer,
    valid_ids: list[int],
    valid_stories: list[str],
    memory_items: list[dict],
    gen_cfg: dict,
    device: torch.device,
    suite: list[dict],
    probe: dict,
    native: dict,
    matched_64: dict,
) -> None:
    v23 = collect_scoreboard_model(
        "v2_3_rope_ctx256",
        config.V2_3_ROPE_BEST_CHECKPOINT,
        active_model,
        valid_ids,
        memory_items,
        gen_cfg,
        device,
        tokenizer,
        suite=suite,
        probe=probe,
        native=native,
        matched_64=matched_64,
    )
    v23["checkpoint_step"] = active_checkpoint_step
    loaded = [
        ("v2", config.V2_BPE_BEST_CHECKPOINT, "v2_bpe"),
        ("v2.1", config.V2_1_CTX256_BEST_CHECKPOINT, "v2_1_ctx256"),
        ("v2.2", config.V2_2_CTX256_BEST_CHECKPOINT, "v2_2_ctx256_longtrain"),
    ]
    sides = {"v2.3": v23}
    models = [("v2.3", active_model)]
    for label, path, experiment in loaded:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {path}. The v2.3 scoreboard does not train earlier models.")
        print(f"Scoring {label} from {path}")
        model, checkpoint = load_model(path, device)
        side = collect_scoreboard_model(
            experiment,
            path,
            model,
            valid_ids,
            memory_items,
            gen_cfg,
            device,
            tokenizer,
        )
        side["checkpoint_step"] = checkpoint.get("step")
        sides[label] = side
        models.append((label, model))
    ordered = [item for label in ("v2", "v2.1", "v2.2", "v2.3") for item in models if item[0] == label]
    print("Generating shared validation-prefix comparisons...")
    stories = compare_real_prefixes(ordered, tokenizer, valid_stories, device)
    success = v23_success(sides["v2"], sides["v2.1"], sides["v2.2"], sides["v2.3"])
    payload = {
        "v2": sides["v2"],
        "v2_1": sides["v2.1"],
        "v2_2": sides["v2.2"],
        "v2_3": sides["v2.3"],
        "stories": stories,
        "success": success,
    }
    json_path = Path(config.STORY_SCOREBOARD_V23_JSON)
    report_path = Path(config.STORY_SCOREBOARD_V23_REPORT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        build_v23_scoreboard(
            sides["v2"],
            sides["v2.1"],
            sides["v2.2"],
            sides["v2.3"],
            stories,
            success,
        ),
        encoding="utf-8",
    )
    print("Saved v2.3 scoreboard:")
    print(report_path.as_posix())


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
    native_validation = {
        "loss": val_loss,
        "perplexity": val_ppl,
        "block_size": model.block_size,
        "batches_evaluated": batches,
        "tokens_evaluated": tokens_evaluated,
    }
    if model.block_size == config.STORY_MEMORY_HIDDEN_TOKENS:
        validation_context_64 = native_validation
    else:
        print("Evaluating matched 64-token validation loss...")
        validation_context_64 = measure_validation(
            model,
            valid_ids,
            config.STORY_MEMORY_HIDDEN_TOKENS,
            device,
        )

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

    memory_items = []
    long_prefix_probe = None
    if tokenizer_type == "bpe":
        print("Selecting long-prefix memory stories...")
        memory_items = select_memory_items(valid_stories, tokenizer)
        print(f"Memory probe stories: {len(memory_items)}")

    print("Running fixed story prompt suite...")
    story_suite = run_story_suite(model, tokenizer, device, gen_cfg)

    if memory_items:
        print("Running long-prefix memory probe...")
        long_prefix_probe = run_memory_probe(model, tokenizer, memory_items, gen_cfg, device)

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
            "experiment": config.EXPERIMENT,
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
        "long_prefix_probe": long_prefix_probe,
        "validation_context_64": validation_context_64,
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

    partner = scoreboard_partner()
    if partner and long_prefix_probe is not None:
        partner_checkpoint, partner_experiment = partner
        if not Path(partner_checkpoint).exists():
            print(f"Scoreboard skipped; missing checkpoint {partner_checkpoint}")
        else:
            print(f"Scoring {partner_experiment} for the story scoreboard...")
            partner_model, _partner_checkpoint_data = load_model(partner_checkpoint, device)
            partner_native = measure_validation(
                partner_model,
                valid_ids,
                partner_model.block_size,
                device,
            )
            if partner_model.block_size == config.STORY_MEMORY_HIDDEN_TOKENS:
                partner_context_64 = partner_native
            else:
                partner_context_64 = measure_validation(
                    partner_model,
                    valid_ids,
                    config.STORY_MEMORY_HIDDEN_TOKENS,
                    device,
                )
            print(f"Running {partner_experiment} story suite and memory probe...")
            partner_suite = run_story_suite(partner_model, tokenizer, device, gen_cfg)
            partner_probe = run_memory_probe(
                partner_model,
                tokenizer,
                memory_items,
                gen_cfg,
                device,
            )
            active_side = scoreboard_side(
                config.EXPERIMENT,
                config.BEST_CHECKPOINT,
                model,
                native_validation,
                validation_context_64,
                story_suite,
                long_prefix_probe,
            )
            partner_side = scoreboard_side(
                partner_experiment,
                partner_checkpoint,
                partner_model,
                partner_native,
                partner_context_64,
                partner_suite,
                partner_probe,
            )
            sides = {
                active_side["experiment"]: active_side,
                partner_side["experiment"]: partner_side,
            }
            long_context = sides.get("v2_1_ctx256") or sides.get("v3_ctx256")
            if long_context is None:
                raise RuntimeError(
                    "The two-model scoreboard needs the context-256 checkpoint beside v2."
                )
            scoreboard = {"v2": sides["v2_bpe"], "v3": long_context}
            scoreboard_json = Path(config.STORY_SCOREBOARD_JSON)
            scoreboard_report = Path(config.STORY_SCOREBOARD_REPORT)
            scoreboard_json.parent.mkdir(parents=True, exist_ok=True)
            scoreboard_json.write_text(json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8")
            scoreboard_report.write_text(
                build_scoreboard(scoreboard["v2"], scoreboard["v3"]),
                encoding="utf-8",
            )
            print("Saved story scoreboard:")
            print(scoreboard_report.as_posix())

    if config.EXPERIMENT == "v2_3_rope_ctx256" and long_prefix_probe is not None:
        write_v23_scoreboard(
            model,
            checkpoint.get("step"),
            tokenizer,
            valid_ids,
            valid_stories,
            memory_items,
            gen_cfg,
            device,
            story_suite,
            long_prefix_probe,
            native_validation,
            validation_context_64,
        )

    if config.EXPERIMENT == "v2_2_ctx256_longtrain" and long_prefix_probe is not None:
        write_family_scoreboard(
            model,
            checkpoint.get("step"),
            tokenizer,
            valid_ids,
            valid_stories,
            memory_items,
            gen_cfg,
            device,
            story_suite,
            long_prefix_probe,
            native_validation,
            validation_context_64,
        )


if __name__ == "__main__":
    main()
