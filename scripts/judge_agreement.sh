
python judge_agreement.py expert \
    --judge_file /workspace/Materials_RL/outputs/human_eval/mixed-10test-0117_DeepSeek-R1-Distill-Qwen-32B_judge_1shot.jsonl \
    --expert_csv "outputs/human_eval/0117-2차평가-예측평가.csv" \
    --label_jsonl outputs/human_eval/mixed-10test-0117.jsonl \
    --id_column "Paper ID" \
    --combine_id_model True \
    --remove_evaluators '["김윤서", "이치훈", "전동원", "홍지훈", "배지수"]'