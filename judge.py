"""
Recipe evaluation using LLM-as-a-Judge with OpenRouter API.
"""

import os
from litellm import completion
import fire
import jsonlines
from tqdm import tqdm


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment variable."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return api_key


USER_PROMPT = """
Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}

Ground Truth Recipe:
{gt_recipe}"""


class RecipeJudge:
    """Judge for evaluating recipe predictions using OpenRouter API."""

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        prompt_filename: str = "prompts/judge.txt",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = open(prompt_filename).read()

        # Set OpenRouter API key
        os.environ["OPENROUTER_API_KEY"] = get_openrouter_api_key()

    def judge(self, item: dict) -> str:
        """Evaluate a single prediction."""
        model = self.model
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_PROMPT.format(
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


def main(
    filename: str,
    model: str = "openai/gpt-5-mini",
    prompt_name: str = "judge",
):
    ds = list(jsonlines.open(filename))
    prompt_filename = f"prompts/{prompt_name}.txt"
    judge = RecipeJudge(model=model, prompt_filename=prompt_filename)

    model_name = model.split("/")[-1]
    output_filename = filename.replace(".jsonl", f"_judged_{model_name}.jsonl")
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
