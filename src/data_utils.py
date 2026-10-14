"""TinyStories text -> one training string and one validation string."""

from datasets import load_dataset


def _take_stories(split: str, limit: int) -> list[str]:
    """Read only the first `limit` stories. The rest of the split is never loaded.

    The slice is taken from the cached dataset. Streaming the parquet shards
    failed on a corrupt Snappy block, while this cached slice still opens.
    """
    dataset = load_dataset("roneneldan/TinyStories", split=f"{split}[:{limit}]")
    stories = []
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"TinyStories {split} row has no text.")
        stories.append(text)
    if len(stories) != limit:
        raise ValueError(f"TinyStories {split} slice has {len(stories)} stories, needed {limit}.")
    return stories


def load_tinystories(num_train: int = 5000, num_valid: int = 500) -> tuple[list[str], list[str]]:
    """Load a small TinyStories slice from Hugging Face.

    The full dataset is millions of stories. A short slice keeps the first
    experiment fast enough to debug.
    """
    if num_train < 1 or num_valid < 1:
        raise ValueError("num_train and num_valid must both be at least 1.")
    return _take_stories("train", num_train), _take_stories("validation", num_valid)


def prepare_text(stories: list[str]) -> str:
    """Join stories into one string, with a blank line between stories."""
    if not stories:
        raise ValueError("Cannot prepare text from an empty story list.")
    return "\n\n".join(story.strip() for story in stories)
