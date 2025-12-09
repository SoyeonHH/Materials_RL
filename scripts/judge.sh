# export CUDA_VISIBLE_DEVICES=2,3

# python judge.py \
#     --filename /workspace/Materials_RL/outputs/test_high_impact/Qwen3-4B-Thinking-2507/prediction.jsonl \
#     --prompt_name judge_1shot \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --tensor_parallel_size 2 \
#     --max_model_len 32768 \
#     --max_tokens 16384

python judge.py \
    --filename /workspace/Materials_RL/outputs/test_high_impact/RetroDFM-R-8B/prediction.jsonl \
    --prompt_name judge_1shot \
    --model openai/gpt-4o-2024-08-06 \
    --max_model_len 16384 \
    --max_tokens 8192