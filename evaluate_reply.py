"""Reply / topic benchmark for chat-style prompts.

TinyStories loss can fall while the model still answers every prompt as a
children's story. This harness measures relevance and story-drift on a fixed
suite, including factual statements like the Lily example.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

import config
from generate import generate_once, load_model
from src.dialogue_data import format_chat_prompt, trim_assistant_reply
from src.tokenizer_utils import load_tokenizer, resolve_tokenizer_type

STORY_MARKERS = (
    r"\bonce upon a time\b",
    r"\bone day\b",
    r"\blittle girl\b",
    r"\blittle boy\b",
    r"\bhappily ever after\b",
    r"\bin a small (village|town|house)\b",
    r"\bthere (lived|was) a\b",
)
STORY_PATTERN = re.compile("|".join(STORY_MARKERS), re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "my",
    "your",
    "i",
    "you",
    "me",
    "it",
    "this",
    "that",
    "do",
    "does",
    "did",
    "can",
    "could",
    "what",
    "how",
    "why",
    "where",
    "when",
    "who",
}


# In-distribution prompts may overlap training. They are not enough to claim generalization.
REPLY_PROMPTS = [
    {"prompt": "hello", "kind": "chat", "split": "trainish", "topic_terms": ["hi", "hello", "help", "today"]},
    {"prompt": "hi", "kind": "chat", "split": "trainish", "topic_terms": ["hi", "hello", "hey"]},
    {"prompt": "how are you", "kind": "chat", "split": "trainish", "topic_terms": ["well", "good", "fine", "you"]},
    {"prompt": "good morning", "kind": "chat", "split": "trainish", "topic_terms": ["morning", "good", "day"]},
    {"prompt": "what is your name", "kind": "chat", "split": "trainish", "topic_terms": ["name", "minigpt", "call"]},
    {"prompt": "tell me a joke", "kind": "chat", "split": "trainish", "topic_terms": ["joke", "why", "because", "funny"]},
    {"prompt": "I am sad", "kind": "chat", "split": "trainish", "topic_terms": ["sad", "sorry", "feel", "talk"]},
    {"prompt": "can you help me", "kind": "chat", "split": "trainish", "topic_terms": ["help", "need", "tell", "try"]},
    {
        "prompt": "Lily's mother is a great musician.",
        "kind": "chat",
        "split": "trainish",
        "topic_terms": ["lily", "mother", "music", "musician", "play", "instrument", "violin", "song"],
    },
    {
        "prompt": "Tom's father is a doctor.",
        "kind": "chat",
        "split": "trainish",
        "topic_terms": ["tom", "father", "doctor", "medicine", "hospital", "work"],
    },
    {
        "prompt": "I like pizza more than pasta.",
        "kind": "chat",
        "split": "trainish",
        "topic_terms": ["pizza", "pasta", "food", "topping", "like"],
    },
    {
        "prompt": "do you like music",
        "kind": "chat",
        "split": "trainish",
        "topic_terms": ["music", "song", "like", "listen"],
    },
    {
        "prompt": "what is 2 plus 2",
        "kind": "chat",
        "split": "trainish",
        "topic_terms": ["4", "four", "plus", "answer"],
    },
    # Held-out prompts: never put these exact lines in training data.
    {
        "prompt": "My mom owns a red car",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["mom", "mother", "car", "red", "drive", "owns"],
    },
    {
        "prompt": "Lily's brother, lost his legs",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["lily", "brother", "legs", "sorry", "hard", "hope"],
    },
    {
        "prompt": "Our neighbor bought a silver van yesterday.",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["neighbor", "van", "silver", "bought", "yesterday", "use"],
    },
    {
        "prompt": "Kate's teacher is a painter.",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["kate", "teacher", "painter", "paint", "art"],
    },
    {
        "prompt": "I prefer rainy days to sunny days.",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["rain", "rainy", "sunny", "prefer", "days"],
    },
    {
        "prompt": "Sam lost his backpack at school.",
        "kind": "chat",
        "split": "held_out",
        "topic_terms": ["sam", "backpack", "school", "lost", "sorry", "last"],
    },
    {"prompt": "Once upon a time", "kind": "story", "split": "control", "topic_terms": ["once", "there", "was", "lived"]},
    {
        "prompt": "The little girl said",
        "kind": "story",
        "split": "control",
        "topic_terms": ["said", "girl", "friend", "help"],
    },
]


def content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def story_drift(text: str, kind: str) -> bool:
    if kind == "story":
        return False
    return bool(STORY_PATTERN.search(text))


def relevance_score(prompt: str, continuation: str, kind: str, topic_terms: list[str]) -> int:
    """0 = off-topic / story-drift, 1 = partial, 2 = on-topic reply.

    Held-out chat prompts require topic-word evidence. A fluent greeting with no
    topic overlap no longer scores as a partial success.
    """
    text = continuation.strip()
    if not text:
        return 0
    drift = story_drift(text, kind)
    lowered = text.lower()
    topic_hits = sum(1 for term in topic_terms if term.lower() in lowered)
    prompt_overlap = len(content_words(prompt) & content_words(text))

    if kind == "story":
        if drift or topic_hits > 0 or prompt_overlap > 0:
            return 2 if len(text.split()) >= 6 else 1
        return 1 if len(text.split()) >= 3 else 0

    if drift and topic_hits == 0:
        return 0
    if topic_hits >= 2 or (topic_hits >= 1 and prompt_overlap >= 1 and not drift):
        return 2
    if topic_hits >= 1:
        return 1
    # Generic greetings or memorized stock lines without topic words count as failures.
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chat reply relevance and story drift.")
    parser.add_argument("--checkpoint", default=config.BEST_CHECKPOINT)
    parser.add_argument("--compare-checkpoint", default=None)
    parser.add_argument("--json", default="experiments/reply_eval.json")
    parser.add_argument("--report", default="experiments/reply_eval.md")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send the bare user text with no User/Assistant wrapper.",
    )
    return parser.parse_args()


def score_model(
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    use_chat_template: bool,
) -> dict:
    samples = []
    for item in REPLY_PROMPTS:
        prompt = item["prompt"]
        model_prompt = prompt if not use_chat_template else format_chat_prompt(prompt)
        sample = generate_once(
            model,
            tokenizer,
            prompt=model_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            device=device,
            stop_on_eos=True,
        )
        text = sample.get("generated_text") or ""
        if text.startswith(model_prompt):
            continuation = text[len(model_prompt) :]
        else:
            continuation = text
        continuation = trim_assistant_reply(continuation)
        drift = story_drift(continuation, item["kind"])
        score = relevance_score(prompt, continuation, item["kind"], item["topic_terms"])
        samples.append(
            {
                "prompt": prompt,
                "kind": item["kind"],
                "split": item.get("split", "trainish"),
                "model_prompt": model_prompt,
                "continuation": continuation,
                "relevance": score,
                "story_drift": drift,
                "stopped_on_eos": sample.get("stopped_on_eos"),
                "tokens_generated": sample.get("tokens_generated"),
            }
        )

    chat = [row for row in samples if row["kind"] == "chat"]
    held = [row for row in samples if row.get("split") == "held_out"]
    story = [row for row in samples if row["kind"] == "story"]
    return {
        "use_chat_template": use_chat_template,
        "samples": samples,
        "relevance_mean": sum(row["relevance"] for row in samples) / len(samples),
        "chat_relevance_mean": sum(row["relevance"] for row in chat) / len(chat),
        "held_out_relevance_mean": (
            sum(row["relevance"] for row in held) / len(held) if held else None
        ),
        "story_relevance_mean": sum(row["relevance"] for row in story) / len(story),
        "story_drift_rate": sum(1 for row in chat if row["story_drift"]) / len(chat),
        "held_out_story_drift_rate": (
            sum(1 for row in held if row["story_drift"]) / len(held) if held else None
        ),
        "chat_prompt_count": len(chat),
        "held_out_prompt_count": len(held),
        "story_prompt_count": len(story),
    }


def side_summary(label: str, result: dict, checkpoint: str) -> dict:
    return {
        "label": label,
        "checkpoint": checkpoint,
        "relevance_mean": result["relevance_mean"],
        "chat_relevance_mean": result["chat_relevance_mean"],
        "held_out_relevance_mean": result["held_out_relevance_mean"],
        "story_relevance_mean": result["story_relevance_mean"],
        "story_drift_rate": result["story_drift_rate"],
        "held_out_story_drift_rate": result["held_out_story_drift_rate"],
        "use_chat_template": result["use_chat_template"],
        "samples": result["samples"],
    }


def build_report(payload: dict) -> str:
    sides = payload["models"]
    headers = " | ".join(side["label"] for side in sides)

    def row(label: str, key: str, digits: int = 3) -> str:
        cells = []
        for side in sides:
            value = side.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        return f"| {label} | {' | '.join(cells)} |"

    blocks = []
    for side in sides:
        lines = [f"### {side['label']}", "", f"Checkpoint: `{side['checkpoint']}`", ""]
        for sample in side["samples"]:
            lines.extend(
                [
                    f"#### `{sample['prompt']}`",
                    "",
                    "```text",
                    sample["continuation"][:600],
                    "```",
                    "",
                    f"- kind: {sample['kind']}",
                    f"- relevance: {sample['relevance']}",
                    f"- story drift: {'yes' if sample['story_drift'] else 'no'}",
                    "",
                ]
            )
        blocks.append("\n".join(lines))

    success = payload.get("success") or {}
    if success:
        verdict = (
            f"Held-out relevance improved: {success.get('held_out_relevance_improved')}. "
            f"Held-out relevance mean: {success.get('held_out_relevance_mean')}. "
            f"Pass held-out relevance: {success.get('pass_held_out_relevance')}. "
            f"Pass held-out drift: {success.get('pass_held_out_drift')}. "
            f"Pass: {success.get('successful')}."
        )
    else:
        verdict = "Single-model baseline. Compare against a second checkpoint to judge improvement."

    return f"""# Reply relevance benchmark

This suite checks whether the model answers chat prompts instead of drifting into TinyStories mode. Grammar alone is not enough. A factual line such as "Lily's mother is a great musician." should stay on music / Lily / mother, not invent a new children's plot.

Decoding: temperature {payload['decoding']['temperature']}, top-k {payload['decoding']['top_k']}, max new tokens {payload['decoding']['max_new_tokens']}, seed {payload['decoding']['seed']}.

| Metric | {headers} |
|{"|".join(["---"] + ["---:"] * len(sides))}|
{row("Held-out relevance mean (0-2)", "held_out_relevance_mean")}
{row("Chat relevance mean (0-2)", "chat_relevance_mean")}
{row("Overall relevance mean (0-2)", "relevance_mean")}
{row("Story-control relevance mean", "story_relevance_mean")}
{row("Held-out story-drift rate", "held_out_story_drift_rate")}
{row("Story-drift rate on chat prompts", "story_drift_rate")}
{row("Chat template", "use_chat_template", 0)}

## Checks

{verdict}

Pass bar: held-out relevance mean at least 1.5, held-out story-drift at most 0.25, and both improved versus the baseline. In-distribution chat scores alone are not enough.

## Samples

{chr(10).join(blocks)}
"""


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_type = resolve_tokenizer_type()
    print("Device:", device.type)
    print("Tokenizer:", tokenizer_type)
    print("Checkpoint:", args.checkpoint)

    tokenizer = load_tokenizer()
    model, checkpoint = load_model(args.checkpoint, device)
    experiment = checkpoint.get("model_config", {}).get("experiment", config.EXPERIMENT)
    def wants_chat_template(name: str | None, path: str) -> bool:
        text = f"{name or ''} {Path(path).name}".lower()
        return "v2_6" in text or "v2_7" in text or "reply" in text

    use_template = (not args.raw_prompt) and wants_chat_template(experiment, args.checkpoint)

    primary = score_model(model, tokenizer, device, args, use_chat_template=use_template)
    sides = [side_summary(experiment or "active", primary, args.checkpoint)]

    success = None
    if args.compare_checkpoint:
        print("Scoring comparison checkpoint:", args.compare_checkpoint)
        other, other_ckpt = load_model(args.compare_checkpoint, device)
        other_exp = other_ckpt.get("model_config", {}).get("experiment", "compare")
        other_template = (not args.raw_prompt) and wants_chat_template(
            other_exp, args.compare_checkpoint
        )
        compared = score_model(other, tokenizer, device, args, use_chat_template=other_template)
        sides.append(side_summary(other_exp or "compare", compared, args.compare_checkpoint))
        # Primary is treated as the candidate when it is v2.6; otherwise compare improves over primary.
        def is_candidate(label: str) -> bool:
            lower = str(label).lower()
            return "v2_7" in lower or "v2_6" in lower or "reply" in lower

        if is_candidate(sides[0]["label"]) and not is_candidate(sides[-1]["label"]):
            candidate, baseline = sides[0], sides[-1]
        elif is_candidate(sides[-1]["label"]) and not is_candidate(sides[0]["label"]):
            candidate, baseline = sides[-1], sides[0]
        else:
            # Newest / first listed reply model versus the other side.
            candidate, baseline = sides[0], sides[-1]
        def num(side: dict, key: str, default: float) -> float:
            value = side.get(key)
            return default if value is None else float(value)

        success = {
            "held_out_relevance_improved": (
                num(candidate, "held_out_relevance_mean", 0)
                > num(baseline, "held_out_relevance_mean", 0)
            ),
            "held_out_drift_improved": (
                num(candidate, "held_out_story_drift_rate", 1)
                <= num(baseline, "held_out_story_drift_rate", 1)
            ),
            "chat_relevance_improved": candidate["chat_relevance_mean"] > baseline["chat_relevance_mean"],
            "story_drift_improved": candidate["story_drift_rate"] < baseline["story_drift_rate"],
            "held_out_relevance_mean": candidate.get("held_out_relevance_mean"),
            "chat_relevance_mean": candidate["chat_relevance_mean"],
            "story_drift_rate": candidate["story_drift_rate"],
            "pass_held_out_relevance": num(candidate, "held_out_relevance_mean", 0) >= 1.5,
            "pass_held_out_drift": num(candidate, "held_out_story_drift_rate", 1) <= 0.25,
        }
        success["successful"] = bool(
            success["held_out_relevance_improved"]
            and success["pass_held_out_relevance"]
            and success["pass_held_out_drift"]
        )

    payload = {
        "models": sides,
        "decoding": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "success": success,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    json_path = Path(args.json)
    report_path = Path(args.report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(payload), encoding="utf-8")
    print("Chat relevance mean:", f"{primary['chat_relevance_mean']:.3f}")
    print("Story-drift rate:", f"{primary['story_drift_rate']:.3f}")
    print("Saved", report_path.as_posix())


if __name__ == "__main__":
    main()
