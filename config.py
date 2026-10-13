# Data
# Downloaded by prepare_tinystories.py. These folders are gitignored.
TRAIN_PATH = "data/raw/tinystories/TinyStories-train.txt"
VALID_PATH = "data/raw/tinystories/TinyStories-valid.txt"

# Explicit tokenizer selection. Do not infer type from filenames.
# "char" = v1 character-level baseline
# "bpe"  = BPE experiments: v2 (context 64), v2.1 and v2.2 (context 256)
TOKENIZER_TYPE = "bpe"

# Active training/eval profile. Artifact paths below follow this value.
# "v1"                    character model, context 64, batch 32, 1000 steps
# "v2_bpe"                BPE model, context 64, batch 32, 1000 steps
# "v2_1_ctx256"           BPE model, context 256, batch 8, 1000 steps
# "v2_2_ctx256_longtrain" same as v2.1, resumed through 3000 total steps
# "v2_3_rope_ctx256"      same budget as v2.2, fresh weights, RoPE on Q and K
EXPERIMENT = "v2_3_rope_ctx256"

CHAR_TOKENIZER_PATH = "data/processed/tokenizer.json"
BPE_TOKENIZER_PATH = "data/processed/bpe_tokenizer.json"
BPE_VOCAB_SIZE = 4096

NUM_TRAIN_STORIES = 5000
NUM_VALID_STORIES = 500

# Model configuration

VOCAB_SIZE = None

# Context length in tokenizer units.
# v1 and v2: 64. v2.1 and v2.2: 256 BPE tokens, enough for one average TinyStory.
# The active profile below overwrites BLOCK_SIZE, BATCH_SIZE, and MAX_STEPS.
V2_BPE_BLOCK_SIZE = 64
V2_BPE_BATCH_SIZE = 32
V2_BPE_MAX_STEPS = 1000
V2_1_BLOCK_SIZE = 256
V2_1_BATCH_SIZE = 8
V2_1_MAX_STEPS = 1000
V2_2_BLOCK_SIZE = 256
V2_2_BATCH_SIZE = 8
V2_2_MAX_STEPS = 3000
V2_3_BLOCK_SIZE = 256
V2_3_BATCH_SIZE = 8
V2_3_MAX_STEPS = 3000

BLOCK_SIZE = V2_BPE_BLOCK_SIZE
EMBEDDING_DIM = 128

NUM_HEADS = 4
NUM_LAYERS = 4

DROPOUT = 0.1


# Training configuration
# BATCH_SIZE is the v2 default until the active profile overwrites it.

BATCH_SIZE = V2_1_BATCH_SIZE
LEARNING_RATE = 3e-4

MAX_STEPS = V2_1_MAX_STEPS
EVAL_INTERVAL = 100
EVAL_BATCHES = 20

GRAD_CLIP = 1.0

# Set to a checkpoint path to continue. A fresh BPE run reuses the saved
# tokenizer when it already exists and does not overwrite that vocabulary.
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

# v2.1 is the 1000-step context-256 model. There is no v3.
# The first training of this checkpoint was saved under the v3_ctx256 filenames.
# Those files stay on disk. Active v2.1 paths are the names below.
V2_1_CTX256_BEST_CHECKPOINT = "checkpoints/v2_1_ctx256_best.pt"
V2_1_CTX256_LATEST_CHECKPOINT = "checkpoints/v2_1_ctx256_latest.pt"
V2_1_CTX256_HISTORY_PATH = "experiments/training_history_v2_1_ctx256.json"
V2_1_CTX256_REPORT_PATH = "experiments/training_report_v2_1_ctx256.md"
V2_1_CTX256_LOSS_CURVE_PATH = "experiments/loss_curve_v2_1_ctx256.png"
V2_1_CTX256_STORY_EVAL_JSON = "experiments/story_eval_v2_1_ctx256.json"
V2_1_CTX256_STORY_EVAL_REPORT = "experiments/story_eval_v2_1_ctx256.md"
V2_1_SOURCE_CHECKPOINT = "checkpoints/v3_ctx256_best.pt"

# v2.2 continues v2.1 to 3000 total steps. It must not overwrite v1, v2, or v2.1.
V2_2_CTX256_BEST_CHECKPOINT = "checkpoints/v2_2_ctx256_longtrain_best.pt"
V2_2_CTX256_LATEST_CHECKPOINT = "checkpoints/v2_2_ctx256_longtrain_latest.pt"
V2_2_CTX256_HISTORY_PATH = "experiments/training_history_v2_2_ctx256_longtrain.json"
V2_2_CTX256_REPORT_PATH = "experiments/training_report_v2_2_ctx256_longtrain.md"
V2_2_CTX256_LOSS_CURVE_PATH = "experiments/loss_curve_v2_2_ctx256_longtrain.png"
V2_2_CTX256_STORY_EVAL_JSON = "experiments/story_eval_v2_2_ctx256_longtrain.json"
V2_2_CTX256_STORY_EVAL_REPORT = "experiments/story_eval_v2_2_ctx256_longtrain.md"

# v2.3 trains a fresh RoPE model for the same 3000 steps. It must not resume v2.2.
V2_3_ROPE_BEST_CHECKPOINT = "checkpoints/v2_3_rope_ctx256_best.pt"
V2_3_ROPE_LATEST_CHECKPOINT = "checkpoints/v2_3_rope_ctx256_latest.pt"
V2_3_ROPE_HISTORY_PATH = "experiments/training_history_v2_3_rope_ctx256.json"
V2_3_ROPE_REPORT_PATH = "experiments/training_report_v2_3_rope_ctx256.md"
V2_3_ROPE_LOSS_CURVE_PATH = "experiments/loss_curve_v2_3_rope_ctx256.png"
V2_3_ROPE_STORY_EVAL_JSON = "experiments/story_eval_v2_3_rope_ctx256.json"
V2_3_ROPE_STORY_EVAL_REPORT = "experiments/story_eval_v2_3_rope_ctx256.md"

TOKENIZER_COMPARISON_REPORT = "experiments/tokenizer_comparison.md"
TOKENIZER_COMPARISON_JSON = "experiments/tokenizer_comparison.json"

# "absolute" keeps the learned position table. "rope" rotates Q and K only.
POSITION_ENCODING = "absolute"

if TOKENIZER_TYPE == "bpe" and EXPERIMENT == "v2_3_rope_ctx256":
    TOKENIZER_PATH = BPE_TOKENIZER_PATH
    BLOCK_SIZE = V2_3_BLOCK_SIZE
    BATCH_SIZE = V2_3_BATCH_SIZE
    MAX_STEPS = V2_3_MAX_STEPS
    POSITION_ENCODING = "rope"
    RESUME_CHECKPOINT = None
    BEST_CHECKPOINT = V2_3_ROPE_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V2_3_ROPE_LATEST_CHECKPOINT
    HISTORY_PATH = V2_3_ROPE_HISTORY_PATH
    REPORT_PATH = V2_3_ROPE_REPORT_PATH
    LOSS_CURVE_PATH = V2_3_ROPE_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V2_3_ROPE_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V2_3_ROPE_STORY_EVAL_REPORT
elif TOKENIZER_TYPE == "bpe" and EXPERIMENT == "v2_2_ctx256_longtrain":
    TOKENIZER_PATH = BPE_TOKENIZER_PATH
    BLOCK_SIZE = V2_2_BLOCK_SIZE
    BATCH_SIZE = V2_2_BATCH_SIZE
    MAX_STEPS = V2_2_MAX_STEPS
    RESUME_CHECKPOINT = V2_1_CTX256_BEST_CHECKPOINT
    BEST_CHECKPOINT = V2_2_CTX256_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V2_2_CTX256_LATEST_CHECKPOINT
    HISTORY_PATH = V2_2_CTX256_HISTORY_PATH
    REPORT_PATH = V2_2_CTX256_REPORT_PATH
    LOSS_CURVE_PATH = V2_2_CTX256_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V2_2_CTX256_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V2_2_CTX256_STORY_EVAL_REPORT
elif TOKENIZER_TYPE == "bpe" and EXPERIMENT == "v2_1_ctx256":
    TOKENIZER_PATH = BPE_TOKENIZER_PATH
    BLOCK_SIZE = V2_1_BLOCK_SIZE
    BATCH_SIZE = V2_1_BATCH_SIZE
    MAX_STEPS = V2_1_MAX_STEPS
    RESUME_CHECKPOINT = None
    BEST_CHECKPOINT = V2_1_CTX256_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V2_1_CTX256_LATEST_CHECKPOINT
    HISTORY_PATH = V2_1_CTX256_HISTORY_PATH
    REPORT_PATH = V2_1_CTX256_REPORT_PATH
    LOSS_CURVE_PATH = V2_1_CTX256_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V2_1_CTX256_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V2_1_CTX256_STORY_EVAL_REPORT
elif TOKENIZER_TYPE == "bpe" and EXPERIMENT == "v2_bpe":
    TOKENIZER_PATH = BPE_TOKENIZER_PATH
    BLOCK_SIZE = V2_BPE_BLOCK_SIZE
    BATCH_SIZE = V2_BPE_BATCH_SIZE
    MAX_STEPS = V2_BPE_MAX_STEPS
    RESUME_CHECKPOINT = None
    BEST_CHECKPOINT = V2_BPE_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V2_BPE_LATEST_CHECKPOINT
    HISTORY_PATH = V2_BPE_HISTORY_PATH
    REPORT_PATH = V2_BPE_REPORT_PATH
    LOSS_CURVE_PATH = V2_BPE_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V2_BPE_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V2_BPE_STORY_EVAL_REPORT
elif TOKENIZER_TYPE == "char" and EXPERIMENT == "v1":
    TOKENIZER_PATH = CHAR_TOKENIZER_PATH
    BLOCK_SIZE = V2_BPE_BLOCK_SIZE
    BATCH_SIZE = V2_BPE_BATCH_SIZE
    MAX_STEPS = V2_BPE_MAX_STEPS
    RESUME_CHECKPOINT = None
    BEST_CHECKPOINT = V1_BEST_CHECKPOINT
    LATEST_CHECKPOINT = V1_LATEST_CHECKPOINT
    HISTORY_PATH = V1_HISTORY_PATH
    REPORT_PATH = V1_REPORT_PATH
    LOSS_CURVE_PATH = V1_LOSS_CURVE_PATH
    STORY_EVAL_JSON = V1_STORY_EVAL_JSON
    STORY_EVAL_REPORT = V1_STORY_EVAL_REPORT
else:
    raise ValueError(
        "Unsupported profile. Use EXPERIMENT 'v1', 'v2_bpe', 'v2_1_ctx256', "
        "'v2_2_ctx256_longtrain', or 'v2_3_rope_ctx256'. "
        f"Got TOKENIZER_TYPE={TOKENIZER_TYPE!r}, EXPERIMENT={EXPERIMENT!r}."
    )

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

# Long-prefix memory probe. The prompt is 128 BPE tokens.
# Terms come from the first 64 tokens, which a 64-token view cannot see.
STORY_MEMORY_PREFIX_TOKENS = 128
STORY_MEMORY_HIDDEN_TOKENS = 64
STORY_MEMORY_STORY_COUNT = 8
STORY_SCOREBOARD_JSON = "experiments/story_scoreboard_v2_v3.json"
STORY_SCOREBOARD_REPORT = "experiments/story_scoreboard_v2_v3.md"
STORY_SCOREBOARD_V2_V21_V22_JSON = "experiments/story_scoreboard_v2_v2_1_v2_2.json"
STORY_SCOREBOARD_V2_V21_V22_REPORT = "experiments/story_scoreboard_v2_v2_1_v2_2.md"
STORY_SCOREBOARD_V23_JSON = "experiments/story_scoreboard_v2_v2_1_v2_2_v2_3.json"
STORY_SCOREBOARD_V23_REPORT = "experiments/story_scoreboard_v2_v2_1_v2_2_v2_3.md"
CONTEXT_UTIL_V23_JSON = "experiments/context_utilization_v2_3.json"
CONTEXT_UTIL_V23_REPORT = "experiments/context_utilization_v2_3.md"

# Same v2.1 checkpoint, three visible-history lengths. No extra training.
CONTEXT_UTIL_JSON = "experiments/context_utilization_v2_1.json"
CONTEXT_UTIL_REPORT = "experiments/context_utilization_v2_1.md"
CONTEXT_UTIL_SEED = 42
CONTEXT_UTIL_SAMPLE_COUNT = 2048
CONTEXT_UTIL_VISIBLE = (64, 128, 256)
CONTEXT_UTIL_EXAMPLE_COUNT = 3
CONTEXT_UTIL_EXAMPLE_NEW_TOKENS = 160
