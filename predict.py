"""
Materials Synthesis Recipe Prediction using Qwen3-4B-Thinking model with vLLM.
Adapted from AlchemyBench/experiment/predict.py
"""

import os
import jsonlines
from tqdm import tqdm
import fire
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Models that support enable_thinking parameter in chat template
QWEN_THINKING_MODELS = ["Qwen3", "Qwen3-4B-Thinking", "Qwen3-8B-Thinking"]


class RecipePredictor:
    """Recipe predictor using vLLM for fast inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        max_new_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.95,
        prompt_filename: str = "prompts/prediction.txt",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        enable_thinking: bool = True,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.prediction_prompt = open(prompt_filename).read()

        # Detect model type for chat template handling
        self.is_qwen_thinking = any(m in model_name for m in QWEN_THINKING_MODELS)
        self.is_deepseek_r1 = "DeepSeek-R1" in model_name

        print(f"Loading model {model_name} with vLLM...")
        print(f"Model type: {'Qwen-Thinking' if self.is_qwen_thinking else 'DeepSeek-R1' if self.is_deepseek_r1 else 'Standard'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=16384,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        print("vLLM model loaded successfully")

    def build_prompt(self, item) -> str:
        """Build prompt from dataset item."""
        contributions = item["contribution"]
        return self.prediction_prompt.format(contributions=contributions)

    def extract_response(self, text: str) -> str:
        """Extract final response, removing <think>...</think> tags."""
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template."""
        messages = [{"role": "user", "content": prompt}]

        if self.is_qwen_thinking:
            # Qwen3-Thinking models support enable_thinking parameter
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            # DeepSeek-R1 and other models use standard chat template
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def predict_batch(self, prompts: list) -> list:
        """Generate predictions for a batch of prompts."""
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        return [self.extract_response(o.outputs[0].text) for o in outputs]

    def predict(self, dataset, batch_size: int = 8):
        """Generate predictions for dataset with batching."""
        prompts, items = [], []

        for item in dataset:
            prompts.append(self.build_prompt(item))
            items.append(item)

            if len(prompts) >= batch_size:
                for it, pred in zip(items, self.predict_batch(prompts)):
                    yield it, pred
                prompts, items = [], []

        if prompts:
            for it, pred in zip(items, self.predict_batch(prompts)):
                yield it, pred


def predict(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    prompt_name: str = "prediction",
    split: str = "test_high_impact",
    max_new_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_samples: int = None,
    output_dir: str = "outputs",
    batch_size: int = 8,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    enable_thinking: bool = True,
):
    """
    Run recipe prediction using Qwen3-4B-Thinking model with vLLM.

    Args:
        model_name: HuggingFace model name
        prompt_name: Name of prompt file (without .txt extension)
        split: Dataset split (train, test_high_impact, test_standard_impact)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_samples: Maximum number of samples to process (None for all)
        output_dir: Directory to save outputs
        batch_size: Batch size for vLLM inference
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        enable_thinking: Enable thinking mode for Qwen3-Thinking models
    """
    from datasets import load_dataset

    print(f"Loading dataset split: {split}")
    ds = load_dataset(
        "parquet",
        data_files=f"open-materials-guide-2024/data/{split}-00000-of-00001.parquet",
        split="train"
    )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"Dataset size: {len(ds)}")

    # Setup output
    model_short_name = model_name.split("/")[-1]
    output_filename = os.path.join(output_dir, split, model_short_name, f"{prompt_name}.jsonl")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Resume from existing progress
    skip = 0
    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        if skip > 0:
            ds = ds.select(range(skip, len(ds)))
            print(f"Resuming from {skip}, {len(ds)} items remaining")

    if len(ds) == 0:
        print("No samples to process")
        return output_filename

    predictor = RecipePredictor(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        prompt_filename=f"prompts/{prompt_name}.txt",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enable_thinking=enable_thinking,
    )

    print(f"Starting predictions, output: {output_filename}")
    print(f"Batch size: {batch_size}, Thinking mode: {enable_thinking}")

    with jsonlines.open(output_filename, "a") as fout:
        for item, prediction in tqdm(predictor.predict(ds, batch_size=batch_size), total=len(ds)):
            fout.write({
                "id": item.get("id"),
                "contribution": item["contribution"],
                "recipe": item["recipe"],
                "prediction": prediction,
            })

    print(f"Predictions saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    fire.Fire(predict)
