"""Download TinyStories train and validation text into data/raw."""

from pathlib import Path

from huggingface_hub import hf_hub_download

import config

REPO_ID = "roneneldan/TinyStories"
FILES = ("TinyStories-train.txt", "TinyStories-valid.txt")


def main():
    raw_dir = Path(config.TRAIN_PATH).parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            local_dir=raw_dir,
        )
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"{name}: {size_mb:.1f} MB -> {path}")


if __name__ == "__main__":
    main()
