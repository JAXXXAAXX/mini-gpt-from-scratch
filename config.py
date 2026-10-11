# Data
# Downloaded by prepare_tinystories.py. These folders are gitignored.
TRAIN_PATH = "data/raw/tinystories/TinyStories-train.txt"
VALID_PATH = "data/raw/tinystories/TinyStories-valid.txt"
TOKENIZER_PATH = "data/processed/tokenizer.json"

# Model configuration

VOCAB_SIZE = None

BLOCK_SIZE = 64
EMBEDDING_DIM = 128

NUM_HEADS = 4
NUM_LAYERS = 2

DROPOUT = 0.1


# Training configuration

BATCH_SIZE = 32
LEARNING_RATE = 3e-4

NUM_EPOCHS = 10

DEVICE = "cpu"