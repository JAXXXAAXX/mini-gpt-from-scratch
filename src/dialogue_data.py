"""Chat-formatted dialogue helpers for reply fine-tuning and the panel.

Hand-written Assistant reply templates were removed. Training chat data should
come from real dialogue corpora (see load_daily_dialogues). Inference still
uses the User / Assistant wrapper and turn trimming below.
"""

from __future__ import annotations

import random


USER = "User"
ASSISTANT = "Assistant"


def format_turn(user: str, assistant: str) -> str:
    return f"{USER}: {user.strip()}\n{ASSISTANT}: {assistant.strip()}"


def format_chat_prompt(user_text: str) -> str:
    """Wrap a bare user line the same way chat fine-tune examples are written."""
    return f"{USER}: {user_text.strip()}\n{ASSISTANT}:"


def trim_assistant_reply(text: str) -> str:
    """Keep one assistant turn. Drop a leaked next User: line if sampling continues."""
    cleaned = text.lstrip("\n")
    marker = f"\n{USER}:"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def dialogue_and_story_documents(
    dialogue_train: list[str],
    story_train: list[str],
    story_ratio: float,
    seed: int = 42,
) -> list[str]:
    """Mix chat docs with a smaller share of TinyStories so fluency is not erased."""
    if not 0.0 <= story_ratio < 1.0:
        raise ValueError("story_ratio must be in [0, 1).")
    rng = random.Random(seed)
    docs = list(dialogue_train)
    if story_ratio > 0 and story_train:
        target_stories = max(1, int(len(dialogue_train) * story_ratio / (1.0 - story_ratio)))
        stories = list(story_train)
        rng.shuffle(stories)
        while len(stories) < target_stories:
            stories.extend(story_train)
        docs.extend(stories[:target_stories])
    rng.shuffle(docs)
    return docs


def load_daily_dialogues(
    num_train: int = 8000,
    num_valid: int = 500,
) -> tuple[list[str], list[str]]:
    """Load real DailyDialog turns. No hand-written reply templates.

    Odd turns are treated as User, even turns as Assistant. Short incomplete
    dialogues are skipped.
    """
    from datasets import load_dataset

    def convert(split: str, limit: int) -> list[str]:
        rows = load_dataset("daily_dialog", split=f"{split}[:{limit * 2}]")
        docs: list[str] = []
        for row in rows:
            turns = [str(turn).strip() for turn in row.get("dialog", []) if str(turn).strip()]
            if len(turns) < 2:
                continue
            lines = []
            for index, turn in enumerate(turns):
                role = USER if index % 2 == 0 else ASSISTANT
                lines.append(f"{role}: {turn}")
            # Keep only dialogues that end on an assistant turn so the LM learns replies.
            if len(lines) % 2 == 1:
                lines = lines[:-1]
            if len(lines) < 2:
                continue
            docs.append("\n".join(lines))
            if len(docs) >= limit:
                break
        if len(docs) < min(limit, 50):
            raise RuntimeError(
                f"DailyDialog {split} produced only {len(docs)} chat documents; needed more."
            )
        return docs

    return convert("train", num_train), convert("validation", num_valid)
