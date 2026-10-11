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
NUM_LAYERS = 4

DROPOUT = 0.1


# Training configuration

BATCH_SIZE = 32
LEARNING_RATE = 3e-4

MAX_STEPS = 1000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20

GRAD_CLIP = 1.0

# Set to a checkpoint path to continue. Fresh runs fit the tokenizer again.
RESUME_CHECKPOINT = None

# Overfit one batch for this many steps when train.py is run with --overfit.
OVERFIT_STEPS = 40

CHECKPOINT_DIR = "checkpoints"
HISTORY_PATH = "experiments/training_history.json"
REPORT_PATH = "experiments/training_report.md"
LOSS_CURVE_PATH = "experiments/loss_curve.png"

# Generation
BEST_CHECKPOINT = "checkpoints/best.pt"
GENERATION_PROMPT = "Once there was"
GENERATION_MAX_NEW_TOKENS = 300
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 20
GENERATION_SEED = 42
GENERATION_SAMPLES_PATH = "experiments/generation_samples.json"
GENERATION_REPORT_PATH = "experiments/generation_report.md"

# Evaluation
FINAL_EVAL_JSON = "experiments/final_evaluation.json"
FINAL_EVAL_REPORT = "experiments/final_report.md"
# None means evaluate every validation batch.
EVAL_MAX_BATCHES = None