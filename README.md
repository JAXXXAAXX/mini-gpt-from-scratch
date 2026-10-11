# Mini GPT from scratch

A small GPT-style language model built in PyTorch, one stage at a time, and trained on a TinyStories subset with a character-level tokenizer.

Detailed results: [experiments/final_report.md](experiments/final_report.md)

## Progress

- Tokenizer ✅
- Dataset pipeline ✅
- Embeddings ✅
- Single-head attention ✅
- Multi-head attention ✅
- Transformer blocks ✅
- Stacked Transformer ✅
- LM head ✅
- Training ✅
- Validation ✅
- Checkpointing ✅
- Generation ✅
- Sampling ✅
- Evaluation ✅
- Final report ✅

## Architecture

```text
Text
→ Character Tokenizer
→ Token + Position Embeddings
→ Transformer Block × 4
→ Final LayerNorm
→ LM Head
→ Vocabulary Logits
```

Default size: embedding 128, 4 heads, 4 layers, context 64, about 0.82M parameters.

## Dataset

TinyStories from Hugging Face. The first experiment uses 5,000 training stories and 500 validation stories. The character vocabulary is saved at `data/processed/tokenizer.json`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional raw text download:

```powershell
python prepare_tinystories.py
```

## Training

```powershell
python train.py
python train.py --overfit
```

Checkpoints are written to `checkpoints/latest.pt` and `checkpoints/best.pt`.

## Generation

```powershell
python generate.py --prompt "Once there was" --temperature 0.8 --top-k 20
python generate.py --run-experiments
```

## Evaluation

```powershell
python evaluate.py
```

This loads `checkpoints/best.pt`, measures validation loss and perplexity, compares baselines, runs generation checks, and writes:

- `experiments/final_evaluation.json`
- `experiments/final_report.md`

## Example generated text

Prompt: `Once there was`

After the 1,000-step training run, sampling produces character-level story-like text. Quality is limited by the short training budget and character tokenizer. See the final report for side-by-side samples.

## Project structure

```text
train.py
generate.py
evaluate.py
config.py
prepare_tinystories.py
src/
  tokenizer.py
  dataset.py
  data_utils.py
  model.py
  trainer.py
  benchmark.py
checkpoints/
experiments/
```

## Current limitations

- Character-level tokenizer
- Context length 64
- Small model and short TinyStories slice
- No pretrained weights, KV cache, top-p, or mixed precision

## Future work

BPE tokenization, larger context and model, longer training, learning-rate schedule, mixed precision, KV cache, top-p sampling, and stronger evaluation metrics.
