"""
Recipe evaluation using LLM-as-a-Judge with OpenRouter API or local vLLM.

Supports quantization for large models to reduce GPU memory:
- AWQ: 4-bit quantization (recommended, fastest)
- GPTQ: 4-bit quantization
- FP8: 8-bit floating point (requires Hopper/Ada GPUs)
- bitsandbytes: 4/8-bit quantization via bitsandbytes

Example usage with quantization:
    python judge.py input.jsonl --model Qwen/Qwen3-Next-80B-A3B-Thinking --quantization awq
"""

import os
import re
from typing import Optional
from litellm import completion
import fire
import jsonlines
from tqdm import tqdm

# Local models that should use vLLM instead of OpenRouter
VLLM_MODELS = [
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "PrimeIntellect/INTELLECT-3-FP8",
]

# Supported quantization methods
SUPPORTED_QUANTIZATION = ["awq", "gptq", "fp8", "bitsandbytes"]


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment variable."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return api_key


def is_vllm_model(model: str) -> bool:
    """Check if model should use vLLM for local inference."""
    return any(m in model for m in VLLM_MODELS)


USER_PROMPT = """
Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}

Ground Truth Recipe:
{gt_recipe}"""


class RecipeJudge:
    """Judge for evaluating recipe predictions using OpenRouter API or local vLLM."""

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        prompt_filename: str = "prompts/judge.txt",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = True,
        max_model_len: int = 16384,
        quantization: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = open(prompt_filename).read()
        self.enable_thinking = enable_thinking
        self.use_vllm = is_vllm_model(model)
        self.quantization = quantization

        if self.use_vllm:
            self._init_vllm(tensor_parallel_size, gpu_memory_utilization, max_model_len, quantization)
        else:
            # Set OpenRouter API key
            print("!!! It is not vLLM model, using OpenRouter API instead !!!")
            os.environ["OPENROUTER_API_KEY"] = get_openrouter_api_key()

    def _init_vllm(
        self,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int = 16384,
        quantization: Optional[str] = None,
    ):
        """Initialize vLLM for local inference.

        Args:
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length (reduces KV cache memory)
            quantization: Quantization method - one of:
                - "awq": 4-bit AWQ quantization (recommended, needs AWQ model)
                - "gptq": 4-bit GPTQ quantization (needs GPTQ model)
                - "fp8": 8-bit FP8 (requires Hopper/Ada GPUs)
                - "bitsandbytes": 4-bit bitsandbytes quantization
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Loading {self.model} with vLLM...")
        print(f"  tensor_parallel_size: {tensor_parallel_size}")
        print(f"  gpu_memory_utilization: {gpu_memory_utilization}")
        print(f"  max_model_len: {max_model_len}")
        if quantization:
            print(f"  quantization: {quantization}")

        # Validate quantization method
        if quantization and quantization not in SUPPORTED_QUANTIZATION:
            raise ValueError(
                f"Unsupported quantization method: {quantization}. "
                f"Supported: {SUPPORTED_QUANTIZATION}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

        # Build LLM kwargs
        llm_kwargs = {
            "model": self.model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
        }

        # Configure quantization
        if quantization == "awq":
            # AWQ quantization - requires AWQ-quantized model
            llm_kwargs["quantization"] = "awq"
            llm_kwargs["dtype"] = "auto"
            print("  Using AWQ 4-bit quantization")
        elif quantization == "gptq":
            # GPTQ quantization - requires GPTQ-quantized model
            llm_kwargs["quantization"] = "gptq"
            llm_kwargs["dtype"] = "auto"
            print("  Using GPTQ 4-bit quantization")
        elif quantization == "fp8":
            # FP8 quantization - requires Hopper/Ada GPUs
            llm_kwargs["quantization"] = "fp8"
            llm_kwargs["dtype"] = "auto"
            print("  Using FP8 quantization (requires H100/L40/RTX 4090)")
        elif quantization == "bitsandbytes":
            # BitsAndBytes 4-bit quantization
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
            llm_kwargs["dtype"] = "auto"
            print("  Using bitsandbytes 4-bit quantization")
        else:
            # No quantization - use bfloat16
            llm_kwargs["dtype"] = "bfloat16"

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.6,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        print("vLLM model loaded successfully")

    def _strip_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from response."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _judge_vllm(self, item: dict) -> str:
        """Evaluate using local vLLM model."""
        user_content = USER_PROMPT.format(
            # objective=item["classification_result"],
            objective=item["contribution"],
            prediction=item["prediction"],
            gt_recipe=item["recipe"],
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template
        if self.enable_thinking:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text

        # Strip thinking tags if present
        response = self._strip_thinking_tags(response)
        return response

    def _judge_openrouter(self, item: dict) -> str:
        """Evaluate using OpenRouter API."""
        model = self.model
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_PROMPT.format(
                # objective=item["classification_result"],
                objective=item["contribution"],
                prediction=item["prediction"],
                gt_recipe=item["recipe"],
            )},
        ]

        response = completion(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"]

    def judge(self, item: dict) -> str:
        """Evaluate a single prediction."""
        if self.use_vllm:
            return self._judge_vllm(item)
        else:
            return self._judge_openrouter(item)


def main(
    filename: str,
    model: str = "openai/gpt-5-mini",
    prompt_name: str = "judge",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enable_thinking: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_model_len: int = 16384,
    quantization: Optional[str] = None,
):
    """Run recipe evaluation using LLM-as-a-Judge.

    Args:
        filename: Input JSONL file with predictions
        model: Model name (OpenRouter format or local vLLM model)
        prompt_name: Prompt template name (from prompts/ directory)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        enable_thinking: Enable thinking tags for compatible models
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        max_model_len: Maximum sequence length (reduces KV cache memory)
        quantization: Quantization method for vLLM models:
            - "awq": 4-bit AWQ (recommended, fastest inference)
            - "gptq": 4-bit GPTQ
            - "fp8": 8-bit FP8 (Hopper/Ada GPUs only)
            - "bitsandbytes": 4-bit via bitsandbytes
    """
    ds = list(jsonlines.open(filename))
    prompt_filename = f"prompts/{prompt_name}.txt"
    judge = RecipeJudge(
        model=model,
        prompt_filename=prompt_filename,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_thinking=enable_thinking,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        quantization=quantization,
    )

    model_name = model.split("/")[-1]
    # Include prompt_name in output filename to avoid conflicts
    output_filename = filename.replace(".jsonl", f"_{model_name}_{prompt_name}.jsonl")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Resume from existing progress
    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        ds = ds[skip:]
        print(f"Skipping {skip} items")

    with jsonlines.open(output_filename, "a") as fout:
        for item in tqdm(ds):
            judge_result = judge.judge(item)
            print(judge_result)
            
            item["judge_result"] = judge_result
            item["judge_model"] = model
            fout.write(item)


if __name__ == "__main__":
    fire.Fire(main)
