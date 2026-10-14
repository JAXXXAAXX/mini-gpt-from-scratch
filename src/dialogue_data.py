"""Chat-formatted dialogue documents for reply fine-tuning.

TinyStories teaches fluent story English. These dialogues teach the model to
answer under a User / Assistant template and to reuse the user's topic words
instead of emitting a memorized stock reply.
"""

from __future__ import annotations

import random


USER = "User"
ASSISTANT = "Assistant"


def format_turn(user: str, assistant: str) -> str:
    return f"{USER}: {user.strip()}\n{ASSISTANT}: {assistant.strip()}"


def format_chat_prompt(user_text: str) -> str:
    """Wrap a bare user line the same way the fine-tune examples are written."""
    return f"{USER}: {user_text.strip()}\n{ASSISTANT}:"


def trim_assistant_reply(text: str) -> str:
    """Keep one assistant turn. Drop a leaked next User: line if sampling continues."""
    cleaned = text.lstrip("\n")
    marker = f"\n{USER}:"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def _ownership_docs(rng: random.Random) -> list[str]:
    people = [
        "mom",
        "dad",
        "sister",
        "brother",
        "friend",
        "uncle",
        "aunt",
        "grandma",
        "teacher",
        "neighbor",
    ]
    owners = ["My", "Our", "Her", "His"]
    verbs = ["owns", "has", "bought", "drives", "got"]
    colors = ["red", "blue", "green", "black", "white", "yellow", "silver"]
    items = [
        ("car", "drive it", "cars"),
        ("bike", "ride it", "bikes"),
        ("truck", "use it", "trucks"),
        ("boat", "take it out", "boats"),
        ("scooter", "ride it", "scooters"),
        ("van", "use it", "vans"),
    ]
    docs = []
    for _ in range(400):
        person = rng.choice(people)
        owner = rng.choice(owners)
        verb = rng.choice(verbs)
        color = rng.choice(colors)
        item, use, plural = rng.choice(items)
        if owner == "My":
            user = f"My {person} {verb} a {color} {item}."
            who = f"your {person}"
        elif owner == "Our":
            user = f"Our {person} {verb} a {color} {item}."
            who = f"your {person}"
        elif owner == "Her":
            user = f"Her {person} {verb} a {color} {item}."
            who = f"her {person}"
        else:
            user = f"His {person} {verb} a {color} {item}."
            who = f"his {person}"
        reply = rng.choice(
            [
                f"A {color} {item} sounds useful. Does {who} {use} often?",
                f"Nice. {color.capitalize()} {plural} stand out. How long has {who} had the {item}?",
                f"Cool. A {color} {item} is a solid choice. Where does {who} keep it?",
            ]
        )
        docs.append(format_turn(user, reply))
    return docs


def _named_fact_docs(rng: random.Random) -> list[str]:
    names = [
        "Lily",
        "Tom",
        "Sara",
        "Anna",
        "Ben",
        "Mia",
        "Noah",
        "Emma",
        "Jack",
        "Olivia",
        "Sam",
        "Kate",
    ]
    relations = ["mother", "father", "brother", "sister", "friend", "teacher"]
    roles = [
        ("musician", "music", "play or sing"),
        ("doctor", "medicine", "work at a hospital"),
        ("teacher", "school", "teach classes"),
        ("chef", "cooking", "make meals"),
        ("painter", "art", "paint pictures"),
        ("runner", "running", "train for races"),
        ("farmer", "farming", "work outside"),
        ("pilot", "flying", "travel for work"),
    ]
    docs = []
    for _ in range(500):
        name = rng.choice(names)
        relation = rng.choice(relations)
        role, topic, action = rng.choice(roles)
        user = f"{name}'s {relation} is a {role}."
        reply = rng.choice(
            [
                f"That is interesting. Does {name} talk about {topic} with their {relation}?",
                f"Nice. Being a {role} takes practice. Does {name} want to {action} too?",
                f"Thanks for sharing. How did {name}'s {relation} get into {topic}?",
            ]
        )
        docs.append(format_turn(user, reply))
    # Include the original demo line, but with several different good replies.
    for reply in (
        "That sounds wonderful. Does Lily like listening to her mother play?",
        "Nice. What kind of music does Lily's mother play?",
        "A musician in the family is special. Does Lily learn from her mother?",
    ):
        docs.append(format_turn("Lily's mother is a great musician.", reply))
    return docs


def _preference_docs(rng: random.Random) -> list[str]:
    pairs = [
        ("pizza", "pasta", "food"),
        ("cats", "dogs", "pets"),
        ("tea", "coffee", "drinks"),
        ("books", "movies", "stories"),
        ("summer", "winter", "seasons"),
        ("soccer", "basketball", "sports"),
        ("drawing", "singing", "hobbies"),
        ("apples", "oranges", "fruit"),
    ]
    docs = []
    for left, right, topic in pairs:
        for frame in (
            f"I like {left} more than {right}.",
            f"I prefer {left} to {right}.",
            f"{left.capitalize()} is better than {right} to me.",
        ):
            reply = rng.choice(
                [
                    f"{left.capitalize()} is a solid choice. What do you like about it compared with {right}?",
                    f"Got it. You prefer {left}. Do you enjoy other {topic} like that too?",
                    f"Fair point. {left.capitalize()} over {right} says a lot about your taste in {topic}.",
                ]
            )
            docs.append(format_turn(frame, reply))
    # Extra random preference lines.
    things = ["rain", "sunny days", "quiet rooms", "loud music", "long walks", "short trips"]
    for _ in range(120):
        a, b = rng.sample(things, 2)
        user = f"I like {a} more than {b}."
        reply = f"You prefer {a} to {b}. What makes {a} better for you?"
        docs.append(format_turn(user, reply))
    return docs


def _hardship_docs(rng: random.Random) -> list[str]:
    """Serious statements need grounded replies, not greetings or jokes."""
    docs = []
    bases = [
        (
            "{name}'s brother lost his {thing}.",
            "I'm sorry to hear that about {name}'s brother. That must be hard. Do you want to talk about how he is doing?",
        ),
        (
            "{name}'s sister is in the hospital.",
            "I'm sorry {name}'s sister is in the hospital. I hope she gets better. Are you staying nearby?",
        ),
        (
            "My friend broke an arm.",
            "Sorry about your friend. A broken arm can heal, but it is still tough. Can you help with daily tasks?",
        ),
        (
            "I failed my test today.",
            "That is disappointing. One test does not define you. Do you want help planning what to study next?",
        ),
        (
            "My dog is sick.",
            "I'm sorry your dog is sick. Has a vet checked on them yet?",
        ),
        (
            "{name} lost a favorite toy.",
            "Losing a favorite toy is upsetting for {name}. Do you know where it was last seen?",
        ),
    ]
    names = ["Lily", "Tom", "Sara", "Ben", "Mia", "Noah"]
    things = ["legs", "job", "backpack", "phone", "bike"]
    for _ in range(220):
        user_t, reply_t = rng.choice(bases)
        name = rng.choice(names)
        thing = rng.choice(things)
        docs.append(
            format_turn(
                user_t.format(name=name, thing=thing),
                reply_t.format(name=name, thing=thing),
            )
        )
    return docs


def _problem_docs(rng: random.Random) -> list[str]:
    docs = []
    problems = [
        ("I lost my keys.", "keys", "That's stressful. Where did you last see your keys?"),
        ("I lost my wallet.", "wallet", "That's stressful. Where did you last see your wallet?"),
        ("I can't find my phone.", "phone", "That's annoying. Where did you last use your phone?"),
        ("My homework is hard.", "homework", "Hard homework is normal. Which subject is giving you trouble?"),
        ("I missed the bus.", "bus", "Missing the bus is frustrating. Can you walk, wait, or call someone?"),
        ("The internet is down.", "internet", "No internet is inconvenient. Have you tried restarting the router?"),
    ]
    for user, topic, reply in problems:
        docs.extend([format_turn(user, reply) for _ in range(15)])
        # Near-paraphrases that keep the topic word in the reply.
        docs.append(
            format_turn(
                user.replace("I ", "Today I "),
                f"Sorry about the {topic}. What have you already tried?",
            )
        )
    return docs


def _greeting_docs() -> list[str]:
    pairs = [
        ("hello", "Hi. How can I help you today?"),
        ("hi", "Hello. What would you like to talk about?"),
        ("good morning", "Good morning. How are you?"),
        ("hey there", "Hey. I'm here if you want to chat."),
        ("how are you", "I'm doing well. How are you?"),
        ("what is your name", "You can call me MiniGPT."),
        ("can you help me", "I can try. What do you need help with?"),
        ("I am sad", "I'm sorry you feel sad. Do you want to talk about it?"),
        ("tell me a joke", "Why did the notebook jump in the pool? It wanted to be a splash page."),
        ("thanks", "You're welcome."),
        ("bye", "Bye. Talk to you later."),
    ]
    # Keep greetings present, but not dominant.
    return [format_turn(user, assistant) for user, assistant in pairs for _ in range(12)]


def _question_docs() -> list[str]:
    pairs = [
        ("what time is it", "I don't have a clock here, but you can check your phone or computer."),
        ("where do you live", "I only exist as a small program on this computer."),
        ("do you like music", "I like talking about music. What kind do you enjoy?"),
        ("what should I eat", "Something simple and filling is a good start. What foods do you have nearby?"),
        ("how do I learn English", "Read short texts every day and practice speaking out loud."),
        ("why is the sky blue", "Sunlight scatters in the air, and blue light spreads more than other colors."),
        ("can you write a short poem", "Sure. Soft rain on the window, quiet thoughts, a calm night."),
        ("what is 2 plus 2", "2 plus 2 is 4."),
        ("what color is a stop sign", "A stop sign is red."),
        ("how many days are in a week", "There are seven days in a week."),
    ]
    return [format_turn(user, assistant) for user, assistant in pairs for _ in range(18)]


def _story_request_docs() -> list[str]:
    pairs = [
        (
            "tell me a short story about a cat",
            "Once there was a small cat who wanted a sunny window. She found one, curled up, and slept happily.",
        ),
        (
            "Once upon a time",
            "Once upon a time, a kind child found a lost kite and returned it to a smiling friend.",
        ),
        (
            "The little girl said",
            "The little girl said she was ready to help, and her friend smiled with relief.",
        ),
        (
            "tell me a short story about a dog",
            "Once there was a dog who loved the park. He ran, played, and went home tired and happy.",
        ),
    ]
    return [format_turn(user, assistant) for user, assistant in pairs for _ in range(12)]


# Exact user lines reserved for evaluate_reply held-out checks. Never train on these.
HELD_OUT_USER_LINES = {
    "My mom owns a red car",
    "Lily's brother, lost his legs",
    "Our neighbor bought a silver van yesterday.",
    "Kate's teacher is a painter.",
    "I prefer rainy days to sunny days.",
    "Sam lost his backpack at school.",
}


def build_synthetic_dialogues(seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (train, valid) chat documents with many compositional topic replies."""
    rng = random.Random(seed)
    train: list[str] = []
    train.extend(_greeting_docs())
    train.extend(_question_docs())
    train.extend(_story_request_docs())
    train.extend(_ownership_docs(rng))
    train.extend(_named_fact_docs(rng))
    train.extend(_preference_docs(rng))
    train.extend(_hardship_docs(rng))
    train.extend(_problem_docs(rng))
    filtered = []
    for doc in train:
        user_line = doc.split("\n", 1)[0]
        if user_line.startswith(f"{USER}: "):
            user_text = user_line[len(f"{USER}: ") :].strip()
            if user_text in HELD_OUT_USER_LINES:
                continue
        filtered.append(doc)
    train = filtered
    rng.shuffle(train)

    # Validation stays small and mostly in-distribution for training loss.
    valid = [
        format_turn("hello", "Hi. How can I help you today?"),
        format_turn("how are you", "I'm doing well. How are you?"),
        format_turn(
            "My sister owns a blue bike.",
            "A blue bike sounds useful. Does your sister ride it often?",
        ),
        format_turn(
            "Tom's father is a doctor.",
            "That is interesting. Does Tom talk about medicine with their father?",
        ),
        format_turn("I like tea more than coffee.", "You prefer tea to coffee. What makes tea better for you?"),
        format_turn(
            "I lost my wallet.",
            "That's stressful. Where did you last see your wallet?",
        ),
        format_turn(
            "My dog is sick.",
            "I'm sorry your dog is sick. Has a vet checked on them yet?",
        ),
        format_turn("what is 2 plus 2", "2 plus 2 is 4."),
        format_turn(
            "tell me a short story about a cat",
            "Once there was a small cat who wanted a sunny window. She found one, curled up, and slept happily.",
        ),
        format_turn(
            "Once upon a time",
            "Once upon a time, a kind child found a lost kite and returned it to a smiling friend.",
        ),
    ]
    return train, valid


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
