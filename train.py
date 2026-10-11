import torch
from torch.utils.data import DataLoader

import config
from src.data_utils import load_tinystories, prepare_text
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import Tokenizer


def encode_validation(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode validation text with the training vocabulary.

    Unknown characters are reported. They are not dropped or replaced,
    because validation must use the same token ids as training.
    """
    try:
        return tokenizer.encode(text)
    except ValueError as exc:
        raise ValueError(
            "Validation text contains characters that are not in the training vocabulary. "
            "The tokenizer is fit only on training stories, so those characters have no token id.\n"
            f"{exc}"
        ) from exc


def main():
    train_stories, valid_stories = load_tinystories(num_train=5000, num_valid=500)
    train_text = prepare_text(train_stories)
    valid_text = prepare_text(valid_stories)

    # Fit on training text only, then reuse that vocabulary for validation.
    tokenizer = Tokenizer()
    tokenizer.fit(train_text)
    tokenizer.save(config.TOKENIZER_PATH)

    train_ids = tokenizer.encode(train_text)
    valid_ids = encode_validation(tokenizer, valid_text)

    train_dataset = TextDataset(train_ids, block_size=config.BLOCK_SIZE)
    valid_dataset = TextDataset(valid_ids, block_size=config.BLOCK_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    train_x, train_y = next(iter(train_loader))
    valid_x, valid_y = next(iter(valid_loader))

    # First window, not a shuffled batch, so the one-character shift is easy to see.
    sample_x, sample_y = train_dataset[0]

    print("training stories:", len(train_stories))
    print("validation stories:", len(valid_stories))
    print("vocab size:", tokenizer.vocab_size)
    print("training token ids:", len(train_ids))
    print("validation token ids:", len(valid_ids))
    print("train batch x:", tuple(train_x.shape))
    print("train batch y:", tuple(train_y.shape))
    print("valid batch x:", tuple(valid_x.shape))
    print("valid batch y:", tuple(valid_y.shape))
    print()
    print("Input:")
    print(tokenizer.decode(sample_x.tolist()))
    print()
    print("Target:")
    print(tokenizer.decode(sample_y.tolist()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.VOCAB_SIZE = tokenizer.vocab_size
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.BLOCK_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
    ).to(device)
    train_x = train_x.to(device)

    positions = torch.arange(train_x.shape[1], device=train_x.device)
    token_emb = model.token_embedding(train_x)
    position_emb = model.position_embedding(positions)
    output = model(train_x)

    print()
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("Token IDs shape:", tuple(train_x.shape))
    print("Token embedding shape:", tuple(token_emb.shape))
    print("Position embedding shape:", tuple(position_emb.shape))
    print("Final embedding output shape:", tuple(output.shape))
    print("Output device:", output.device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    if torch.cuda.is_available() and output.device.type != "cuda":
        raise RuntimeError(f"Expected the embedding output on CUDA, got {output.device}.")


if __name__ == "__main__":
    main()
