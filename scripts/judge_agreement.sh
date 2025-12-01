# python judge_agreement.py expert \
#     --judge_file SynthEval/humaneval-0117/humaneval-0117/data/judge/gpt-4o-2024-08-06/mixed-10test-0117-0210-v9.jsonl \
#     --expert_csv "outputs/human_eval/0117-2차평가-예측평가.csv" \
#     --label_jsonl outputs/human_eval/mixed-10test-0117.jsonl \
#     --id_column "Paper ID" \
#     --combine_id_model False \
#     --remove_evaluators '["김윤서", "이치훈", "전동원", "홍지훈", "배지수"]'


python judge_agreement.py expert \
    --judge_file outputs/human_eval/mixed-10test-0117_Qwen3-30B-A3B-Thinking-2507_judge.jsonl \
    --expert_csv "outputs/human_eval/0117-2차평가-예측평가.csv" \
    --label_jsonl outputs/human_eval/mixed-10test-0117.jsonl \
    --id_column "Paper ID" \
    --combine_id_model True \
    --remove_evaluators '["김윤서", "이치훈", "전동원", "홍지훈", "배지수"]'