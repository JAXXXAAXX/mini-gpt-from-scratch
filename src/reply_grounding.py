"""Prompt-grounded reply scoring and decoding.

These helpers do not add hand-written chat templates. They only:
1. measure whether a reply invents closed attributes missing from the user text
2. bias sampling toward tokens that already appear in the user prompt
3. rerank a few samples by faithfulness to the user's words and polarity
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F


COLORS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "yellow",
    "silver",
    "gold",
    "purple",
    "orange",
    "pink",
    "brown",
    "gray",
    "grey",
}

BODY_PARTS = {
    "arm",
    "arms",
    "leg",
    "legs",
    "hand",
    "hands",
    "foot",
    "feet",
    "head",
    "neck",
    "back",
    "knee",
    "knees",
    "finger",
    "fingers",
}

CLOSED_ATTRIBUTES = COLORS | BODY_PARTS | {
    "sick",
    "vet",
    "vaccine",
    "fever",
    "cold",
    "flu",
}

# TinyStories-heavy names and family roles often leak into chat replies.
STORY_NAMES = {
    "lily",
    "tom",
    "sam",
    "kate",
    "anna",
    "lucy",
    "jack",
    "ben",
    "sara",
    "sarah",
    "max",
    "mia",
    "emma",
    "noah",
    "oliver",
    "sophia",
}

FAMILY_ROLES = {
    "brother",
    "sister",
    "uncle",
    "aunt",
    "cousin",
    "grandma",
    "grandfather",
    "grandmother",
    "grandpa",
}

# Career / hobby topic families that training templates over-apply.
TOPIC_FAMILIES = {
    "music": {"music", "musician", "song", "sing", "violin", "piano", "guitar", "instrument"},
    "medicine": {"doctor", "medicine", "nurse", "clinic", "vet"},
    "art": {"art", "paint", "painter", "drawing", "draw"},
}

POLARITY_GROUPS = (
    (
        {"upset", "sad", "angry", "mad", "worried", "scared", "afraid", "hurt", "broke", "broken", "lost"},
        {
            "happy",
            "glad",
            "excited",
            "cheerful",
            "joyful",
            "fun",
            "friendly",
            "smile",
            "smiling",
            "nice",
            "play",
            "playing",
        },
    ),
)

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
    "her",
    "his",
    "their",
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
    "about",
    "very",
    "has",
    "have",
    "had",
    "get",
    "into",
    "our",
    "their",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def content_words(text: str) -> set[str]:
    return {word for word in words(text) if word not in STOPWORDS and len(word) > 1}


def distinctive_words(text: str) -> set[str]:
    """Longer content words that should usually reappear in a grounded reply."""
    return {word for word in content_words(text) if len(word) >= 3}


def invented_attributes(prompt: str, reply: str) -> list[str]:
    """Closed attributes that appear in the reply but not in the user prompt."""
    allowed = set(words(prompt))
    found = []
    seen = set()
    for word in words(reply):
        if word in CLOSED_ATTRIBUTES and word not in allowed and word not in seen:
            seen.add(word)
            found.append(word)
    return found


def invented_entities(prompt: str, reply: str) -> list[str]:
    """Story names / family roles that appear in the reply but not the prompt."""
    allowed = set(words(prompt))
    blocked = STORY_NAMES | FAMILY_ROLES
    found = []
    seen = set()
    for word in words(reply):
        if word in blocked and word not in allowed and word not in seen:
            seen.add(word)
            found.append(word)
    return found


def topic_family_leaks(prompt: str, reply: str) -> list[str]:
    """Topic families present in the reply but unsupported by the user text."""
    prompt_words = set(words(prompt))
    reply_words = set(words(reply))
    leaks = []
    for family, members in TOPIC_FAMILIES.items():
        if reply_words & members and not (prompt_words & members):
            leaks.append(family)
    return leaks


def polarity_conflicts(prompt: str, reply: str) -> list[str]:
    """Negative prompt cues answered with positive filler (or the reverse)."""
    prompt_words = set(words(prompt))
    reply_words = set(words(reply))
    conflicts = []
    for negative, positive in POLARITY_GROUPS:
        if not positive:
            continue
        if prompt_words & negative and reply_words & positive:
            conflicts.extend(sorted(reply_words & positive))
        if prompt_words & positive and reply_words & negative:
            conflicts.extend(sorted(reply_words & negative))
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for word in conflicts:
        if word not in seen:
            seen.add(word)
            ordered.append(word)
    return ordered


def faithfulness_score(prompt: str, reply: str) -> float:
    """Higher is better. Penalize invents; reward reuse of the user's words."""
    text = reply.strip()
    if not text:
        return -10.0
    prompt_content = content_words(prompt)
    reply_content = content_words(reply)
    distinctive = distinctive_words(prompt)
    overlap = prompt_content & reply_content
    distinctive_hits = distinctive & reply_content

    invented = invented_attributes(prompt, reply)
    entities = invented_entities(prompt, reply)
    leaks = topic_family_leaks(prompt, reply)
    polarity = polarity_conflicts(prompt, reply)

    score = float(len(overlap))
    if distinctive:
        score += 3.0 * (len(distinctive_hits) / len(distinctive))
        if not distinctive_hits:
            score -= 5.0
    score -= 3.0 * len(invented)
    score -= 4.0 * len(entities)
    score -= 3.5 * len(leaks)
    score -= 4.0 * len(polarity)
    # Soft preference for short, single-turn answers.
    score -= max(0, len(text.split()) - 40) * 0.05
    return score


def is_faithful(prompt: str, reply: str) -> bool:
    return (
        not invented_attributes(prompt, reply)
        and not invented_entities(prompt, reply)
        and not polarity_conflicts(prompt, reply)
    )


def _encode_piece(tokenizer, piece: str) -> list[int]:
    try:
        encoded = tokenizer.encode(piece, add_bos=False, add_eos=False)
    except TypeError:
        encoded = tokenizer.encode(piece)
    return [int(token_id) for token_id in encoded]


def _token_ids_for_words(tokenizer, word_list: set[str]) -> set[int]:
    """Single-token spellings only (safe to hard-mask)."""
    ids: set[int] = set()
    for word in word_list:
        for piece in (f" {word}", f" {word.capitalize()}", word, word.capitalize()):
            encoded = _encode_piece(tokenizer, piece)
            if len(encoded) == 1:
                ids.add(encoded[0])
    return ids


def _blocked_word_set(user_text: str) -> set[str]:
    allowed = set(words(user_text))
    blocked_words = {word for word in CLOSED_ATTRIBUTES if word not in allowed}
    blocked_words |= {word for word in (STORY_NAMES | FAMILY_ROLES) if word not in allowed}
    prompt_words = set(words(user_text))
    for negative, positive in POLARITY_GROUPS:
        if prompt_words & negative and positive:
            blocked_words |= positive
    for members in TOPIC_FAMILIES.values():
        if not (prompt_words & members):
            blocked_words |= members
    return blocked_words


def _banned_sequences(tokenizer, word_list: set[str]) -> list[list[int]]:
    """Full token sequences for blocked words, including multi-token BPE paths."""
    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for word in word_list:
        for piece in (f" {word}", f" {word.capitalize()}", word, word.capitalize()):
            encoded = _encode_piece(tokenizer, piece)
            if not encoded:
                continue
            key = tuple(encoded)
            if key not in seen:
                seen.add(key)
                sequences.append(encoded)
    return sequences


def _next_tokens_to_ban(generated_ids: list[int], banned_sequences: list[list[int]]) -> set[int]:
    """If the suffix matches a banned prefix, ban the next id in that sequence."""
    ban: set[int] = set()
    for sequence in banned_sequences:
        if len(sequence) == 1:
            ban.add(sequence[0])
            continue
        max_prefix = min(len(sequence) - 1, len(generated_ids))
        for prefix_len in range(1, max_prefix + 1):
            if generated_ids[-prefix_len:] == sequence[:prefix_len]:
                ban.add(sequence[prefix_len])
    return ban


def prompt_boost_token_ids(tokenizer, user_text: str) -> set[int]:
    """Token ids that should be encouraged because they come from the user text."""
    ids: set[int] = set()
    ids.update(_encode_piece(tokenizer, user_text))
    ids.update(_token_ids_for_words(tokenizer, content_words(user_text)))
    return ids


def blocked_attribute_token_ids(tokenizer, user_text: str) -> set[int]:
    """Single-token ids for words absent from the prompt."""
    return _token_ids_for_words(tokenizer, _blocked_word_set(user_text))


def scrub_unfaithful(prompt: str, reply: str) -> str:
    """Drop or clean sentences that still invent blocked attributes/entities."""
    banned = set(invented_attributes(prompt, reply)) | set(invented_entities(prompt, reply))
    if not banned:
        return reply.strip()
    parts = re.split(r"(?<=[.!?])\s+", reply.strip())
    kept: list[str] = []
    for part in parts:
        part_words = set(words(part))
        if not (part_words & banned):
            kept.append(part)
            continue
        cleaned = part
        for word in banned:
            cleaned = re.sub(rf"(?i)\b{re.escape(word)}\b", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned).strip(" ,;:-")
        if len(content_words(cleaned)) >= 3:
            kept.append(cleaned)
    text = " ".join(kept).strip()
    return text or reply.strip()


def _candidate_rank(prompt: str, reply: str) -> tuple:
    return (
        -len(invented_attributes(prompt, reply)),
        -len(invented_entities(prompt, reply)),
        -len(polarity_conflicts(prompt, reply)),
        -len(topic_family_leaks(prompt, reply)),
        faithfulness_score(prompt, reply),
    )


@torch.inference_mode()
def generate_grounded(
    model,
    tokenizer,
    prompt_ids: list[int],
    user_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
    eos_token_id: int | None,
    candidates: int = 12,
    boost: float = 3.0,
    block_penalty: float = 10.0,
) -> str:
    """Sample several replies with prompt grounding and keep the most faithful one."""
    del block_penalty  # kept for call-site compatibility; bans are hard masks now
    if candidates < 1:
        raise ValueError("candidates must be at least 1.")
    boost_ids = prompt_boost_token_ids(tokenizer, user_text)
    blocked_words = _blocked_word_set(user_text)
    block_ids = _token_ids_for_words(tokenizer, blocked_words)
    banned_sequences = _banned_sequences(tokenizer, blocked_words)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    scored: list[tuple[tuple, str]] = []
    for _ in range(candidates):
        generated = prompt_tensor
        for _step in range(max_new_tokens):
            context = generated[:, -model.block_size :]
            logits = model(context)[:, -1, :] / temperature
            vocab = logits.size(-1)
            if boost_ids:
                index = torch.tensor(
                    [token_id for token_id in sorted(boost_ids) if 0 <= token_id < vocab],
                    dtype=torch.long,
                    device=device,
                )
                if index.numel():
                    logits[:, index] += boost
            ban_ids = set(block_ids)
            ban_ids |= _next_tokens_to_ban(generated[0].tolist(), banned_sequences)
            if ban_ids:
                index = torch.tensor(
                    [token_id for token_id in sorted(ban_ids) if 0 <= token_id < vocab],
                    dtype=torch.long,
                    device=device,
                )
                if index.numel():
                    logits[:, index] = float("-inf")
            if top_k is not None:
                keep = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, keep)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            # If masking emptied the distribution, fall back to unmasked top-k.
            if torch.isneginf(logits).all():
                logits = model(context)[:, -1, :] / temperature
                if top_k is not None:
                    keep = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, keep)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
        continuation_ids = generated[0].tolist()[len(prompt_ids) :]
        continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        continuation = scrub_unfaithful(user_text, continuation)
        scored.append((_candidate_rank(user_text, continuation), continuation))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]

