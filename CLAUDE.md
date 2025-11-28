# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Materials_RL is a materials synthesis recipe prediction and evaluation system. It uses LLMs to generate synthesis recipes from key research contributions, then evaluates predictions against ground truth recipes using an LLM-as-a-Judge approach.

The project is based on the Open Materials Guide (OMG) dataset containing 17,000+ expert-verified synthesis recipes.

## Common Commands

### Recipe Prediction with vLLM

```bash
# Run prediction on test set (uses Qwen3-4B-Thinking by default)
python predict.py --split test_high_impact

# Full prediction with custom parameters
python predict.py \
    --model_name "Qwen/Qwen3-4B-Thinking-2507" \
    --split test_high_impact \
    --batch_size 8 \
    --max_new_tokens 8192 \
    --temperature 0.6 \
    --enable_thinking True

# Process limited samples for testing
python predict.py --split test_high_impact --max_samples 10
```

### Evaluation with LLM-as-a-Judge

```bash
# Judge predictions using GPT-4o
python judge.py outputs/test_high_impact/Qwen3-4B-Thinking-2507/prediction.jsonl \
    --model gpt-4o-2024-08-06 \
    --prompt_name judge
```

## Architecture

### Pipeline Flow

1. **Input**: Dataset item with `contribution` field (key research contributions)
2. **Prediction**: `predict.py` uses vLLM to generate synthesis recipes
3. **Evaluation**: `judge.py` compares predictions against ground truth using LLM-as-a-Judge

### Key Components

- **`predict.py`**: VLLMPredictor class for batched inference with Qwen3-Thinking models
- **`judge.py`**: RecipeJudge class (extends RecipePredictor) for evaluation scoring
- **`prompts/`**: Prompt templates
  - `prediction.txt`: Main synthesis recipe generation prompt
  - `judge.txt`: Evaluation rubric (7 criteria, 1-5 scale)
  - `rag.txt`: RAG-augmented prediction prompt

### Dataset Structure

Located in `open-materials-guide-2024/data/`:
- `train-00000-of-00001.parquet` (16,026 recipes)
- `test_standard_impact-00000-of-00001.parquet` (1,472 recipes)
- `test_high_impact-00000-of-00001.parquet` (169 recipes)

### Output Format

Predictions saved to `outputs/{split}/{model_name}/{prompt_name}.jsonl` with fields:
- `id`, `contribution`, `recipe` (ground truth), `prediction`

Judged outputs append `_judged_{model}.jsonl` with `judge_result` scores.

## Dependencies

- vLLM for fast inference
- transformers for tokenization
- litellm/openai for judge API calls
- datasets for loading parquet data
- fire for CLI argument parsing
- jsonlines for output I/O

## Related Work

This project is related to AlchemyBench (https://github.com/iknow-lab/AlchemyBench). See the paper: "Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge" (arXiv:2502.16457).
