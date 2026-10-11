# Data
# Downloaded by prepare_tinystories.py. These folders are gitignored.
TRAIN_PATH = "data/raw/tinystories/TinyStories-train.txt"
VALID_PATH = "data/raw/tinystories/TinyStories-valid.txt"

# Explicit tokenizer selection. Do not infer type from filenames.
# "char" = v1 character-level baseline
# "bpe"  = v2 BPE / subword experiment
TOKENIZER_TYPE = "bpe"

CHAR_TOKENIZER_PATH = "data/processed/tokenizer.json"
BPE_TOKENIZER_PATH = "data/processed/bpe_tokenizer.json"
BPE_VOCAB_SIZE = 4096

NUM_TRAIN_STORIES = 5000
NUM_VALID_STORIES = 500

# Model configuration

VOCAB_SIZE = None

# Context length in tokenizer units.
# v1: 64 character tokens. v2: 64 BPE tokens. Same number, different text span.
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

# Preserved v1 character-model artifacts (do not overwrite when training BPE).
V1_BEST_CHECKPOINT = "checkpoints/best.pt"
V1_LATEST_CHECKPOINT = "checkpoints/latest.pt"
V1_HISTORY_PATH = "experiments/training_history.json"
V1_REPORT_PATH = "experiments/training_report.md"
V1_LOSS_CURVE_PATH = "experiments/loss_curve.png"
V1_STORY_EVAL_JSON = "experiments/story_eval_v1.json"
V1_STORY_EVAL_REPORT = "experiments/story_eval_v1.md"

# Versioned v2 BPE artifacts
V2_BPE_BEST_CHECKPOINT = "checkpoints/v2_bpe_best.pt"
V2_BPE_LATEST_CHECKPOINT = "checkpoints/v2_bpe_latest.pt"
V2_BPE_HISTORY_PATH = "experiments/training_history_v2_bpe.json"
V2_BPE_REPORT_PATH = "experiments/training_report_v2_bpe.md"
V2_BPE_LOSS_CURVE_PATH = "experiments/loss_curve_v2_bpe.png"
V2_BPE_STORY_EVAL_JSON = "experiments/story_eval_v2_bpe.json"
V2_BPE_STORY_EVAL_REPORT = "experiments/story_eval_v2_bpe.md"

TOKENIZER_COMPARISON_REPORT = "experiments/tokenizer_comparison.md"
TOKENIZER_COMPARISON_JSON = "experiments/tokenizer_comparison.json"

if TOKENIZER_TYPE == "bpe":
    TOKENIZER_PATH = BPE_TOKENIZER_PATH
    BEST_CHECKPOINT = V2_BPE_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V2_BPE_LATEST_CHECKPOINT
    HISTORY_PATH = V2_BPE_HISTORY_PATH
    REPORT_PATH = V2_BPE_REPORT_PATH
    LOSS_CURVE_PATH = V2_BPE_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V2_BPE_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V2_BPE_STORY_EVAL_REPORT
elif TOKENIZER_TYPE == "char":
    TOKENIZER_PATH = CHAR_TOKENIZER_PATH
    BEST_CHECKPOINT = V1_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V1_LATEST_CHECKPOINT
    HISTORY_PATH = V1_HISTORY_PATH
    REPORT_PATH = V1_REPORT_PATH
    LOSS_CURVE_PATH = V1_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V1_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V1_STORY_EVAL_REPORT
else:
    raise ValueError(f"TOKENIZER_TYPE must be 'char' or 'bpe', got {TOKENIZER_TYPE!r}.")

# Generation
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

# Story generation evaluation harness
STORY_EVAL_MAX_NEW_TOKENS = 300
STORY_EVAL_TEMPERATURE = 0.8
STORY_EVAL_TOP_K = 20
STORY_EVAL_SEED = 42
