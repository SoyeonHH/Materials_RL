"""
Extract average scores from judged recipe prediction results.
"""

import json
import re
import fire
import jsonlines


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text containing ```json ... ``` blocks."""
    pattern = r"```json\s*(\{[^`]+\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


SCORE_KEYS = [
    "materials_appropriateness_score",
    "equipment_appropriateness_score",
    "procedure_completeness_score",
    "procedure_similarity_score",
    "procedure_feasibility_score",
    "characterization_appropriateness_score",
    "characterization_similarity_score",
    "overall_score",
]


def compute_average_scores(filename: str, output_filename: str = None) -> dict:
    """
    Compute average scores from judged results.

    Args:
        filename: Path to judged JSONL file
        output_filename: Path to save JSON output (default: same as input with _scores.json suffix)

    Returns:
        Dictionary with average scores per criterion
    """
    if output_filename is None:
        output_filename = filename.replace(".jsonl", "_scores.json")

    scores = {key: [] for key in SCORE_KEYS}
    failed_count = 0
    total_count = 0

    with jsonlines.open(filename) as reader:
        for item in reader:
            total_count += 1
            judge_result = item.get("judge_result", "")
            extracted = extract_json_from_text(judge_result)

            if extracted is None:
                failed_count += 1
                continue

            for key in SCORE_KEYS:
                if key in extracted:
                    scores[key].append(float(extracted[key]))

    # Compute averages
    avg_scores = {}
    for key in SCORE_KEYS:
        if scores[key]:
            avg_scores[key] = round(sum(scores[key]) / len(scores[key]), 3)
        else:
            avg_scores[key] = None

    result = {
        "total_samples": total_count,
        "parsed_samples": total_count - failed_count,
        "failed_samples": failed_count,
        "average_scores": avg_scores,
    }

    with open(output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scores saved to {output_filename}")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    fire.Fire(compute_average_scores)
