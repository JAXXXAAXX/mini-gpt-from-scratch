"""Character vs BPE tokenizer benchmark and controlled comparison report.

Measures compression and unknown-token rates on the same TinyStories slice,
records parameter impact, and joins available v1 / v2 story-eval results.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import config
from evaluate import load_json
from evaluate_story import STORY_PROMPTS
from src.bpe_tokenizer import BPETokenizer
from src.data_utils import load_tinystories
from src.tokenizer import Tokenizer
from src.tokenizer_utils import encode_stories, parameter_breakdown, story_token_counts

EXAMPLE_TEXTS = [
    "Once there was a little girl.",
    "The dog was very happy because he found a ball.",
    "One day the children went outside to play.",
]


def _median(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_tokenizer(
    name: str,
    tokenizer,
    stories: list[str],
    *,
    use_story_boundaries: bool,
) -> dict:
    counts = story_token_counts(tokenizer, stories, use_story_boundaries=use_story_boundaries)
    ids = encode_stories(tokenizer, stories, use_story_boundaries=use_story_boundaries)
    total_chars = sum(len(story) for story in stories)
    total_tokens = sum(counts)
    unk_count = 0
    if isinstance(tokenizer, BPETokenizer):
        unk_count = tokenizer.count_unk(ids)
    return {
        "name": name,
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", name),
        "vocab_size": tokenizer.vocab_size,
        "story_count": len(counts),
        "total_token_count": total_tokens,
        "total_character_count": total_chars,
        "average_tokens_per_story": _mean(counts),
        "median_tokens_per_story": _median(counts),
        "average_characters_per_token": total_chars / total_tokens if total_tokens else None,
        "unknown_token_count": unk_count,
        "unknown_token_rate": unk_count / len(ids) if ids else 0.0,
    }


def example_rows(char_tokenizer: Tokenizer, bpe_tokenizer: BPETokenizer) -> list[dict]:
    rows = []
    for text in EXAMPLE_TEXTS:
        char_ids = char_tokenizer.encode(text)
        bpe_ids = bpe_tokenizer.encode(text)
        rows.append(
            {
                "raw_text": text,
                "character_token_count": len(char_ids),
                "bpe_token_count": len(bpe_ids),
                "bpe_token_pieces": bpe_tokenizer.token_pieces(text),
            }
        )
    return rows


def prompt_unk_stats(bpe_tokenizer: BPETokenizer, prompts: list[str]) -> dict:
    total = 0
    unk = 0
    for prompt in prompts:
        ids = bpe_tokenizer.encode(prompt)
        total += len(ids)
        unk += bpe_tokenizer.count_unk(ids)
    return {
        "prompt_count": len(prompts),
        "token_count": total,
        "unknown_token_count": unk,
        "unknown_token_rate": unk / total if total else 0.0,
    }


def suite_metric_mean(story_eval: dict | None, key: str) -> float | None:
    if not story_eval:
        return None
    values = []
    for item in story_eval.get("story_suite", []):
        value = item.get("diagnostics", {}).get(key)
        if value is not None:
            values.append(value)
    return _mean(values)


def with_derived_chars_per_sec(story_eval: dict | None, chars_per_token: float | None) -> dict | None:
    """Fill characters/sec from tokens/sec when older v1 artifacts lack the field."""
    if not story_eval:
        return None
    payload = json.loads(json.dumps(story_eval))
    performance = payload.setdefault("performance", {})
    if (
        performance.get("training_characters_per_second") is None
        and performance.get("training_tokens_per_second") is not None
        and chars_per_token is not None
    ):
        performance["training_characters_per_second"] = (
            performance["training_tokens_per_second"] * chars_per_token
        )
    if (
        performance.get("mean_suite_characters_per_second") is None
        and performance.get("mean_suite_tokens_per_second") is not None
        and chars_per_token is not None
    ):
        performance["mean_suite_characters_per_second"] = (
            performance["mean_suite_tokens_per_second"] * chars_per_token
        )
    return payload


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(result: dict) -> str:
    char = result["character"]
    bpe = result["bpe"]
    compression = result["compression_ratio_char_tokens_over_bpe_tokens"]
    params_char = result["parameters"]["character"]
    params_bpe = result["parameters"]["bpe"]
    v1 = result.get("story_eval_v1") or {}
    v2 = result.get("story_eval_v2_bpe") or {}
    examples = "\n\n".join(
        (
            f"### `{item['raw_text']}`\n\n"
            f"- Character tokens: {item['character_token_count']}\n"
            f"- BPE tokens: {item['bpe_token_count']}\n"
            f"- BPE pieces: `{item['bpe_token_pieces']}`"
        )
        for item in result["examples"]
    )
    side_by_side = []
    v1_suite = {item["prompt"]: item for item in v1.get("story_suite", [])}
    v2_suite = {item["prompt"]: item for item in v2.get("story_suite", [])}
    for prompt in STORY_PROMPTS:
        left = v1_suite.get(prompt, {}).get("generated_text", "_unavailable_")
        right = v2_suite.get(prompt, {}).get("generated_text", "_unavailable_")
        side_by_side.append(
            f"### Prompt `{prompt}`\n\n"
            f"**Character v1**\n\n```text\n{left}\n```\n\n"
            f"**BPE v2**\n\n```text\n{right}\n```"
        )

    return f"""# Character vs BPE Tokenizer

## Configuration

| Setting | Character v1 | BPE v2 |
|---|---:|---:|
| Tokenizer type | char | bpe |
| Configured vocab size | fitted from train chars | {config.BPE_VOCAB_SIZE} |
| Train stories | {config.NUM_TRAIN_STORIES} | {config.NUM_TRAIN_STORIES} |
| Valid stories | {config.NUM_VALID_STORIES} | {config.NUM_VALID_STORIES} |
| Block size | {config.BLOCK_SIZE} char tokens | {config.BLOCK_SIZE} BPE tokens |
| Embedding dim / heads / layers | {config.EMBEDDING_DIM} / {config.NUM_HEADS} / {config.NUM_LAYERS} | same |
| Max steps | {config.MAX_STEPS} | {config.MAX_STEPS} |
| Story boundaries | blank-line join | `<bos> story <eos>` |

Important: `{config.BLOCK_SIZE}` character tokens and `{config.BLOCK_SIZE}` BPE tokens cover very different amounts of text. This experiment keeps the numeric block size fixed on purpose.

## Vocabulary Size

| Tokenizer | Vocab Size |
|---|---:|
| Character | {char['vocab_size']} |
| BPE | {bpe['vocab_size']} |

## Token Compression

Compression ratio = character-token count / BPE-token count on the same stories.

| Metric | Character | BPE |
|---|---:|---:|
| Total tokens (train stories) | {fmt(char['total_token_count'], 0)} | {fmt(bpe['total_token_count'], 0)} |
| Avg tokens / story | {fmt(char['average_tokens_per_story'], 1)} | {fmt(bpe['average_tokens_per_story'], 1)} |
| Median tokens / story | {fmt(char['median_tokens_per_story'], 1)} | {fmt(bpe['median_tokens_per_story'], 1)} |
| Avg characters / token | {fmt(char['average_characters_per_token'], 3)} | {fmt(bpe['average_characters_per_token'], 3)} |
| Compression ratio | 1.000 | {fmt(compression, 3)} |

## Example Tokenization

{examples}

## Unknown Token Rate

| Split / Set | Unknown tokens | Unknown rate |
|---|---:|---:|
| BPE train stories | {fmt(bpe['unknown_token_count'], 0)} | {fmt(bpe['unknown_token_rate'], 6)} |
| BPE validation stories | {fmt(result['bpe_validation']['unknown_token_count'], 0)} | {fmt(result['bpe_validation']['unknown_token_rate'], 6)} |
| Fixed story prompts | {fmt(result['prompt_unk']['unknown_token_count'], 0)} | {fmt(result['prompt_unk']['unknown_token_rate'], 6)} |

Desired behavior for normal English TinyStories text is near-zero `<unk>`.

## Model Parameter Impact

| Component | Character v1 | BPE v2 |
|---|---:|---:|
| Token embedding | {fmt(params_char['token_embedding_params'], 0)} | {fmt(params_bpe['token_embedding_params'], 0)} |
| Transformer (+ final LN) | {fmt(params_char['transformer_params'], 0)} | {fmt(params_bpe['transformer_params'], 0)} |
| LM head | {fmt(params_char['lm_head_params'], 0)} | {fmt(params_bpe['lm_head_params'], 0)} |
| Total | {fmt(params_char['total_params'], 0)} | {fmt(params_bpe['total_params'], 0)} |

The Transformer body is unchanged. Most of the parameter increase comes from the larger embedding table and LM head.

## Training Performance

| Metric | Character v1 | BPE v2 |
|---|---:|---:|
| Val Loss | {fmt((v1.get('validation') or {}).get('loss'))} | {fmt((v2.get('validation') or {}).get('loss'))} |
| Perplexity | {fmt((v1.get('validation') or {}).get('perplexity'), 2)} | {fmt((v2.get('validation') or {}).get('perplexity'), 2)} |
| Train tokens/sec | {fmt((v1.get('performance') or {}).get('training_tokens_per_second'), 0)} | {fmt((v2.get('performance') or {}).get('training_tokens_per_second'), 0)} |
| Train characters/sec | {fmt((v1.get('performance') or {}).get('training_characters_per_second'), 0)} | {fmt((v2.get('performance') or {}).get('training_characters_per_second'), 0)} |
| Gen tokens/sec | {fmt((v1.get('performance') or {}).get('mean_suite_tokens_per_second'), 1)} | {fmt((v2.get('performance') or {}).get('mean_suite_tokens_per_second'), 1)} |
| Gen characters/sec | {fmt((v1.get('performance') or {}).get('mean_suite_characters_per_second'), 1)} | {fmt((v2.get('performance') or {}).get('mean_suite_characters_per_second'), 1)} |

Tokens/sec is not a fair efficiency comparison across tokenizers. Prefer characters/sec.

## Story Evaluation

| Metric | Character v1 | BPE v2 |
|---|---:|---:|
| Vocabulary Size | {fmt(char['vocab_size'], 0)} | {fmt(bpe['vocab_size'], 0)} |
| Avg Tokens / Story | {fmt(char['average_tokens_per_story'], 1)} | {fmt(bpe['average_tokens_per_story'], 1)} |
| Avg Characters / Token | {fmt(char['average_characters_per_token'], 3)} | {fmt(bpe['average_characters_per_token'], 3)} |
| Model Parameters | {fmt(params_char['total_params'], 0)} | {fmt(params_bpe['total_params'], 0)} |
| Val Loss | {fmt((v1.get('validation') or {}).get('loss'))} | {fmt((v2.get('validation') or {}).get('loss'))} |
| Perplexity | {fmt((v1.get('validation') or {}).get('perplexity'), 2)} | {fmt((v2.get('validation') or {}).get('perplexity'), 2)} |
| Train Tokens/sec | {fmt((v1.get('performance') or {}).get('training_tokens_per_second'), 0)} | {fmt((v2.get('performance') or {}).get('training_tokens_per_second'), 0)} |
| Gen Tokens/sec | {fmt((v1.get('performance') or {}).get('mean_suite_tokens_per_second'), 1)} | {fmt((v2.get('performance') or {}).get('mean_suite_tokens_per_second'), 1)} |
| Distinct-2 | {fmt(suite_metric_mean(v1, 'distinct_2'), 3)} | {fmt(suite_metric_mean(v2, 'distinct_2'), 3)} |
| Repeated 3-gram Rate | {fmt(suite_metric_mean(v1, 'repeated_3gram_rate'), 3)} | {fmt(suite_metric_mean(v2, 'repeated_3gram_rate'), 3)} |

### Side-by-side generations

Same prompts and primary decoding settings (`max_new_tokens=300`, `temperature=0.8`, `top_k=20`, `seed=42`) where available. Manual quality is left to the rubric fields in the story-eval JSON files.

{chr(10).join(side_by_side)}

## Controlled Comparison Notes

- Dataset size, Transformer body, optimizer, learning rate, batch size, and step budget are held constant.
- Only the tokenizer (and the resulting vocab-linked embedding / LM-head sizes) changed for v2.
- Do not claim BPE is better from perplexity alone; character and BPE losses are not on the same token units.
- Manual rubric scores remain `null` until human review.
- Preserved v1 artifacts: character tokenizer, `checkpoints/best.pt`, `experiments/story_eval_v1.*`, and architecture compute benchmarks.
"""


def main() -> None:
    print("Loading TinyStories train/valid slices...")
    train_stories, valid_stories = load_tinystories(
        num_train=config.NUM_TRAIN_STORIES,
        num_valid=config.NUM_VALID_STORIES,
    )

    char_path = Path(config.CHAR_TOKENIZER_PATH)
    if not char_path.exists():
        raise FileNotFoundError(
            f"Character tokenizer missing at {char_path}. Train v1 first or restore it."
        )
    char_tokenizer = Tokenizer.load(str(char_path))

    bpe_path = Path(config.BPE_TOKENIZER_PATH)
    if bpe_path.exists():
        print(f"Loading existing BPE tokenizer from {bpe_path.as_posix()}")
        bpe_tokenizer = BPETokenizer.load(str(bpe_path))
    else:
        print("Training BPE tokenizer on train stories only...")
        bpe_tokenizer = BPETokenizer()
        bpe_tokenizer.train(train_stories, vocab_size=config.BPE_VOCAB_SIZE)
        bpe_tokenizer.save(str(bpe_path))
        print(f"Saved {bpe_path.as_posix()}")

    char_stats = summarize_tokenizer(
        "character", char_tokenizer, train_stories, use_story_boundaries=False
    )
    bpe_stats = summarize_tokenizer(
        "bpe", bpe_tokenizer, train_stories, use_story_boundaries=True
    )
    bpe_valid = summarize_tokenizer(
        "bpe_validation", bpe_tokenizer, valid_stories, use_story_boundaries=True
    )
    compression = (
        char_stats["total_token_count"] / bpe_stats["total_token_count"]
        if bpe_stats["total_token_count"]
        else None
    )

    result = {
        "configuration": {
            "num_train_stories": config.NUM_TRAIN_STORIES,
            "num_valid_stories": config.NUM_VALID_STORIES,
            "bpe_vocab_size_target": config.BPE_VOCAB_SIZE,
            "block_size": config.BLOCK_SIZE,
            "embedding_dim": config.EMBEDDING_DIM,
            "num_heads": config.NUM_HEADS,
            "num_layers": config.NUM_LAYERS,
            "max_steps": config.MAX_STEPS,
        },
        "character": char_stats,
        "bpe": bpe_stats,
        "bpe_validation": bpe_valid,
        "compression_ratio_char_tokens_over_bpe_tokens": compression,
        "examples": example_rows(char_tokenizer, bpe_tokenizer),
        "prompt_unk": prompt_unk_stats(bpe_tokenizer, STORY_PROMPTS),
        "parameters": {
            "character": parameter_breakdown(char_tokenizer.vocab_size),
            "bpe": parameter_breakdown(bpe_tokenizer.vocab_size),
        },
        "story_eval_v1": with_derived_chars_per_sec(
            load_json(config.V1_STORY_EVAL_JSON),
            char_stats.get("average_characters_per_token"),
        ),
        "story_eval_v2_bpe": with_derived_chars_per_sec(
            load_json(config.V2_BPE_STORY_EVAL_JSON),
            bpe_stats.get("average_characters_per_token"),
        ),
    }

    json_path = Path(config.TOKENIZER_COMPARISON_JSON)
    report_path = Path(config.TOKENIZER_COMPARISON_REPORT)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_markdown(result), encoding="utf-8")

    print("Compression ratio (char tokens / BPE tokens):", fmt(compression, 3))
    print("BPE vocab size:", bpe_tokenizer.vocab_size)
    print("Saved:", json_path.as_posix())
    print("Saved:", report_path.as_posix())


if __name__ == "__main__":
    main()
