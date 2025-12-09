"""
Measure agreement between judge models on recipe evaluation.

Computes inter-rater agreement metrics including:
- Pearson correlation
- Spearman correlation
- Cohen's Kappa (discretized scores)
- Mean Absolute Difference
- T-test for score distribution comparison

Supports:
- Judge vs Judge comparison
- Judge vs Expert (human) comparison
- Self-agreement (same model multiple runs)
"""

import json
import os
import re
import unicodedata
import fire
import jsonlines
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def normalize_path(path: str) -> str:
    """
    Normalize file path to handle NFD/NFC unicode differences.
    macOS uses NFD, while most other systems use NFC.
    """
    directory = os.path.dirname(path)
    filename = os.path.basename(path)

    # If file exists with given path, return as-is
    if os.path.exists(path):
        return path

    # Try to find file with normalized name
    if directory and os.path.exists(directory):
        for f in os.listdir(directory):
            if unicodedata.normalize('NFC', f) == unicodedata.normalize('NFC', filename):
                return os.path.join(directory, f)

    return path


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

# Mapping from judge score keys to expert CSV column names
# Note: Judge's overall_score maps to "Overall" (not "Overall Score") to match notebook behavior
JUDGE_TO_EXPERT_KEYS = {
    "materials_appropriateness_score": "Material_Appropriateness",
    "equipment_appropriateness_score": "Equipment_Appropriateness",
    "procedure_completeness_score": "Procedure_Completeness",
    "procedure_similarity_score": "Procedure_Similarity",
    "procedure_feasibility_score": "Procedure_Feasibility",
    "characterization_appropriateness_score": "Characterization_Appropriateness",
    "characterization_similarity_score": "Characterization_Similarity",
    "overall_score": "Overall",
}

# Columns used for computing average overall score
EXPERT_SCORE_COLS = [
    "Material_Appropriateness",
    "Equipment_Appropriateness",
    "Procedure_Completeness",
    "Procedure_Similarity",
    "Procedure_Feasibility",
    "Characterization_Appropriateness",
    "Characterization_Similarity",
]


def load_scores(filename: str) -> dict[str, dict]:
    """
    Load scores from judged JSONL file.

    Returns:
        Dictionary mapping sample id to scores dict
    """
    scores = {}
    with jsonlines.open(filename) as reader:
        for item in reader:
            sample_id = item.get("id")
            if sample_id is None:
                continue

            judge_result = item.get("judge_result", "")
            extracted = extract_json_from_text(judge_result)

            if extracted is not None:
                scores[sample_id] = {
                    key: float(extracted[key])
                    for key in SCORE_KEYS
                    if key in extracted
                }

    return scores


def discretize_score(score: float, bins: int = 5) -> int:
    """Discretize score to integer bins for Cohen's Kappa."""
    return int(min(max(round(score), 1), bins))


def compute_agreement(
    file1: str,
    file2: str,
    output_filename: str = None,
) -> dict:
    """
    Compute agreement metrics between two judge result files.

    Args:
        file1: Path to first judged JSONL file
        file2: Path to second judged JSONL file
        output_filename: Path to save JSON output (optional)

    Returns:
        Dictionary with agreement metrics
    """
    scores1 = load_scores(file1)
    scores2 = load_scores(file2)

    # Find common samples
    common_ids = set(scores1.keys()) & set(scores2.keys())
    print(f"Found {len(common_ids)} common samples")
    print(f"  File 1: {len(scores1)} samples")
    print(f"  File 2: {len(scores2)} samples")

    if len(common_ids) == 0:
        print("No common samples found!")
        return {"error": "No common samples"}

    # Collect paired scores
    paired_scores = {key: {"judge1": [], "judge2": []} for key in SCORE_KEYS}

    for sample_id in common_ids:
        s1 = scores1[sample_id]
        s2 = scores2[sample_id]

        for key in SCORE_KEYS:
            if key in s1 and key in s2:
                paired_scores[key]["judge1"].append(s1[key])
                paired_scores[key]["judge2"].append(s2[key])

    # Compute metrics for each score key
    metrics = {}

    for key in SCORE_KEYS:
        j1 = np.array(paired_scores[key]["judge1"])
        j2 = np.array(paired_scores[key]["judge2"])

        if len(j1) < 2:
            metrics[key] = {"error": "Insufficient samples"}
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(j1, j2)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(j1, j2)

        # Mean Absolute Difference
        mad = np.mean(np.abs(j1 - j2))

        # Cohen's Kappa (discretized)
        j1_discrete = [discretize_score(s) for s in j1]
        j2_discrete = [discretize_score(s) for s in j2]
        kappa = cohen_kappa_score(j1_discrete, j2_discrete, weights="quadratic")

        # Exact agreement rate
        exact_agreement = np.mean(np.array(j1_discrete) == np.array(j2_discrete))

        # Within 0.5 agreement rate
        within_half = np.mean(np.abs(j1 - j2) <= 0.5)

        metrics[key] = {
            "n_samples": len(j1),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "cohens_kappa": round(kappa, 4),
            "mean_abs_diff": round(mad, 4),
            "exact_agreement": round(exact_agreement, 4),
            "within_half_agreement": round(within_half, 4),
            "judge1_mean": round(np.mean(j1), 4),
            "judge2_mean": round(np.mean(j2), 4),
        }

    # Compute overall agreement (average across all score keys)
    valid_metrics = [m for m in metrics.values() if "error" not in m]

    if valid_metrics:
        overall = {
            "n_samples": valid_metrics[0]["n_samples"],
            "avg_pearson_r": round(np.mean([m["pearson_r"] for m in valid_metrics]), 4),
            "avg_spearman_r": round(np.mean([m["spearman_r"] for m in valid_metrics]), 4),
            "avg_cohens_kappa": round(np.mean([m["cohens_kappa"] for m in valid_metrics]), 4),
            "avg_mean_abs_diff": round(np.mean([m["mean_abs_diff"] for m in valid_metrics]), 4),
            "avg_exact_agreement": round(np.mean([m["exact_agreement"] for m in valid_metrics]), 4),
            "avg_within_half_agreement": round(np.mean([m["within_half_agreement"] for m in valid_metrics]), 4),
        }
    else:
        overall = {"error": "No valid metrics computed"}

    result = {
        "file1": file1,
        "file2": file2,
        "common_samples": len(common_ids),
        "overall_agreement": overall,
        "per_criterion": metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL AGREEMENT")
    print("=" * 60)
    if "error" not in overall:
        print(f"  Pearson r:           {overall['avg_pearson_r']:.4f}")
        print(f"  Spearman r:          {overall['avg_spearman_r']:.4f}")
        print(f"  Cohen's Kappa:       {overall['avg_cohens_kappa']:.4f}")
        print(f"  Mean Abs Diff:       {overall['avg_mean_abs_diff']:.4f}")
        print(f"  Exact Agreement:     {overall['avg_exact_agreement']:.2%}")
        print(f"  Within 0.5:          {overall['avg_within_half_agreement']:.2%}")

    print("\n" + "=" * 60)
    print("PER-CRITERION AGREEMENT")
    print("=" * 60)
    print(f"{'Criterion':<40} {'Pearson':>8} {'Kappa':>8} {'MAD':>8}")
    print("-" * 60)
    for key in SCORE_KEYS:
        m = metrics[key]
        if "error" not in m:
            short_key = key.replace("_score", "").replace("_", " ")
            print(f"{short_key:<40} {m['pearson_r']:>8.4f} {m['cohens_kappa']:>8.4f} {m['mean_abs_diff']:>8.4f}")

    # Save results
    if output_filename is None:
        # Generate default output filename
        import os
        base1 = os.path.basename(file1).replace(".jsonl", "")
        base2 = os.path.basename(file2).replace(".jsonl", "")
        output_filename = f"agreement_{base1}_vs_{base2}.json"

    with open(output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_filename}")

    return result


def load_judge_scores_as_df(filename: str, combine_id_model: bool = False) -> pd.DataFrame:
    """
    Load judge scores from JSONL file as a DataFrame.

    Args:
        filename: Path to judged JSONL file
        combine_id_model: If True, combine 'id' and 'model' fields to create
                          composite ID (e.g., "paper-04-o1-mini")

    Returns:
        DataFrame with sample id as index and scores as columns
    """
    records = []
    with jsonlines.open(filename) as reader:
        for item in reader:
            sample_id = item.get("id")
            if sample_id is None:
                continue

            # Optionally combine id and model for matching with expert data
            if combine_id_model and "model" in item:
                sample_id = f"{sample_id}-{item['model']}"

            judge_result = item.get("judge_result", "")
            extracted = extract_json_from_text(judge_result)

            if extracted is not None:
                record = {"id": sample_id}
                for key in SCORE_KEYS:
                    if key in extracted:
                        # Convert to expert-style column name
                        col_name = JUDGE_TO_EXPERT_KEYS.get(key, key)
                        record[col_name] = float(extracted[key])
                records.append(record)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df.set_index("id", inplace=True)

        # Compute average overall score from sub-scores
        valid_cols = [c for c in EXPERT_SCORE_COLS if c in df.columns]
        if valid_cols:
            df["Overall Score AVG"] = df[valid_cols].mean(axis=1)

    return df


def load_expert_scores(
    expert_csv: str,
    label_jsonl: str = None,
    id_column: str = "Paper ID",
    remove_evaluators: list[str] = None,
) -> pd.DataFrame:
    """
    Load expert evaluation scores from CSV file.

    Args:
        expert_csv: Path to expert evaluation CSV file
        label_jsonl: Path to label JSONL file for ID mapping (optional)
        id_column: Column name containing sample IDs
        remove_evaluators: List of evaluator names to exclude

    Returns:
        DataFrame with sample id as index and averaged scores per sample
    """
    # Normalize path to handle NFD/NFC unicode differences
    expert_csv = normalize_path(expert_csv)
    df = pd.read_csv(expert_csv)

    # Remove explanation columns
    explanation_cols = [c for c in df.columns if 'explanation' in c.lower() or "Unnamed" in c]
    df = df.drop(explanation_cols, axis=1, errors='ignore')

    # Remove specified evaluators
    if remove_evaluators and "Evaluation" in df.columns:
        df = df[~df["Evaluation"].isin(remove_evaluators)]

    # Create ID mapping from label file if provided
    if label_jsonl and os.path.exists(label_jsonl):
        label_df = pd.read_json(label_jsonl, lines=True)
        if "model" in label_df.columns:
            label_df["combined_id"] = [
                f"{id}-{model}" for id, model in zip(label_df["id"], label_df["model"])
            ]
            row2model = {
                f"Row-{i:02d}": label_df.loc[i, "combined_id"]
                for i in range(len(label_df))
            }
            if id_column in df.columns:
                df[id_column] = df[id_column].apply(lambda x: row2model.get(x, x))

    # Set index
    if id_column in df.columns:
        df.set_index(id_column, inplace=True)

    # Average scores per sample (across multiple evaluators) and round to match notebook behavior
    df_experts = df.groupby(df.index).mean(numeric_only=True).round()

    # Compute average overall score from sub-scores (after rounding)
    valid_cols = [c for c in EXPERT_SCORE_COLS if c in df_experts.columns]
    if valid_cols:
        df_experts["Overall Score AVG"] = df_experts[valid_cols].mean(axis=1)

    return df_experts


def compute_expert_agreement(
    judge_file: str,
    expert_csv: str,
    label_jsonl: str = None,
    id_column: str = "Paper ID",
    remove_evaluators: list[str] = None,
    output_filename: str = None,
    use_avg_overall: bool = True,
    combine_id_model: bool = True,
) -> dict:
    """
    Compute agreement between judge model and expert evaluations.

    Args:
        judge_file: Path to judged JSONL file
        expert_csv: Path to expert evaluation CSV file
        label_jsonl: Path to label JSONL file for ID mapping (optional)
        id_column: Column name containing sample IDs in expert CSV
        remove_evaluators: List of evaluator names to exclude
        output_filename: Path to save JSON output (optional)
        use_avg_overall: Use averaged overall score instead of direct overall score
        combine_id_model: If True, combine 'id' and 'model' fields in judge file
                          to match expert IDs (e.g., "paper-04-o1-mini")

    Returns:
        Dictionary with agreement metrics
    """
    # Load data
    df_judge = load_judge_scores_as_df(judge_file, combine_id_model=combine_id_model)
    df_experts = load_expert_scores(
        expert_csv,
        label_jsonl=label_jsonl,
        id_column=id_column,
        remove_evaluators=remove_evaluators,
    )

    # Find common samples
    common_ids = set(df_judge.index) & set(df_experts.index)
    print(f"Found {len(common_ids)} common samples")
    print(f"  Judge: {len(df_judge)} samples")
    print(f"  Expert: {len(df_experts)} samples")

    if len(common_ids) == 0:
        print("No common samples found!")
        return {"error": "No common samples"}

    # Filter to common samples
    df_judge = df_judge.loc[list(common_ids)]
    df_experts = df_experts.loc[list(common_ids)]

    # Compute metrics for each score key
    metrics = {}

    # Build list of (judge_col, expert_col, display_name) tuples
    score_pairs = []
    for judge_key, col_name in JUDGE_TO_EXPERT_KEYS.items():
        if col_name == "Overall":
            # Judge's "Overall" compares with Expert's "Overall Score AVG" (notebook behavior)
            if use_avg_overall and "Overall" in df_judge.columns and "Overall Score AVG" in df_experts.columns:
                score_pairs.append(("Overall", "Overall Score AVG", "Overall (Judge) vs Overall Score AVG (Expert)"))
        else:
            # For other columns, compare same column names
            if col_name in df_judge.columns and col_name in df_experts.columns:
                score_pairs.append((col_name, col_name, col_name))

    # Also add Judge's computed "Overall Score AVG" vs Expert's "Overall Score AVG" for comparison
    if "Overall Score AVG" in df_judge.columns and "Overall Score AVG" in df_experts.columns:
        score_pairs.append(("Overall Score AVG", "Overall Score AVG", "Overall Score AVG"))

    for judge_col, expert_col, display_name in score_pairs:
        judge_scores = df_judge[judge_col].dropna()
        expert_scores = df_experts[expert_col].dropna()

        # Align indices
        common_idx = judge_scores.index.intersection(expert_scores.index)
        if len(common_idx) < 2:
            metrics[display_name] = {"error": "Insufficient samples"}
            continue

        j = judge_scores.loc[common_idx].values
        e = expert_scores.loc[common_idx].values

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(j, e)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(j, e)

        # T-test
        ttest_stat, ttest_p = stats.ttest_ind(j, e)

        # Mean Absolute Difference
        mad = np.mean(np.abs(j - e))

        # Cohen's Kappa (discretized)
        j_discrete = [discretize_score(s) for s in j]
        e_discrete = [discretize_score(s) for s in e]
        try:
            kappa = cohen_kappa_score(j_discrete, e_discrete, weights="quadratic")
        except ValueError:
            kappa = np.nan

        metrics[display_name] = {
            "n_samples": len(common_idx),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "ttest_stat": round(ttest_stat, 4),
            "ttest_p": round(ttest_p, 4),
            "cohens_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            "mean_abs_diff": round(mad, 4),
            "judge_mean": round(np.mean(j), 4),
            "expert_mean": round(np.mean(e), 4),
        }

    # Compute overall agreement
    valid_metrics = [m for m in metrics.values() if "error" not in m]

    if valid_metrics:
        overall = {
            "n_samples": valid_metrics[0]["n_samples"],
            "avg_pearson_r": round(np.mean([m["pearson_r"] for m in valid_metrics]), 4),
            "avg_spearman_r": round(np.mean([m["spearman_r"] for m in valid_metrics]), 4),
            "avg_cohens_kappa": round(np.nanmean([m["cohens_kappa"] for m in valid_metrics if m["cohens_kappa"] is not None]), 4),
            "avg_mean_abs_diff": round(np.mean([m["mean_abs_diff"] for m in valid_metrics]), 4),
        }
    else:
        overall = {"error": "No valid metrics computed"}

    result = {
        "judge_file": judge_file,
        "expert_csv": expert_csv,
        "common_samples": len(common_ids),
        "overall_agreement": overall,
        "per_criterion": metrics,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("JUDGE vs EXPERT AGREEMENT")
    print("=" * 70)
    if "error" not in overall:
        print(f"  Pearson r:           {overall['avg_pearson_r']:.4f}")
        print(f"  Spearman r:          {overall['avg_spearman_r']:.4f}")
        print(f"  Cohen's Kappa:       {overall['avg_cohens_kappa']:.4f}")
        print(f"  Mean Abs Diff:       {overall['avg_mean_abs_diff']:.4f}")

    print("\n" + "=" * 70)
    print("PER-CRITERION AGREEMENT")
    print("=" * 70)
    print(f"{'Criterion':<35} {'Pearson':>10} {'Spearman':>10} {'t-test':>10} {'MAD':>8}")
    print("-" * 70)
    for col, m in metrics.items():
        if "error" not in m:
            short_col = col.replace("_", " ")
            p_str = f"{m['pearson_r']:.2f}({m['pearson_p']:.2f})"
            s_str = f"{m['spearman_r']:.2f}({m['spearman_p']:.2f})"
            t_str = f"{m['ttest_stat']:.2f}({m['ttest_p']:.2f})"
            print(f"{short_col:<35} {p_str:>10} {s_str:>10} {t_str:>10} {m['mean_abs_diff']:>8.2f}")

    # Save results
    if output_filename is None:
        judge_base = os.path.basename(judge_file).replace(".jsonl", "")
        expert_base = os.path.basename(expert_csv).replace(".csv", "")
        output_filename = f"agreement_{judge_base}_vs_{expert_base}.json"

    with open(output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_filename}")

    return result


def batch_expert_agreement(
    judge_dir: str,
    expert_csv: str,
    judge_pattern: str = "*.jsonl",
    label_jsonl: str = None,
    id_column: str = "Paper ID",
    remove_evaluators: list[str] = None,
    output_filename: str = None,
) -> dict:
    """
    Compute agreement for multiple judge files against expert evaluations.

    Args:
        judge_dir: Directory containing judged JSONL files
        expert_csv: Path to expert evaluation CSV file
        judge_pattern: Glob pattern for judge files
        label_jsonl: Path to label JSONL file for ID mapping
        id_column: Column name containing sample IDs
        remove_evaluators: List of evaluator names to exclude
        output_filename: Path to save JSON output

    Returns:
        Dictionary with agreement metrics for each judge model
    """
    import glob

    judge_files = glob.glob(os.path.join(judge_dir, judge_pattern))
    if not judge_files:
        judge_files = glob.glob(os.path.join(judge_dir, "**", judge_pattern), recursive=True)

    print(f"Found {len(judge_files)} judge files")

    results = {}

    for judge_file in sorted(judge_files):
        # Extract model name from path
        model_name = os.path.basename(os.path.dirname(judge_file))
        if not model_name or model_name == judge_dir:
            model_name = os.path.basename(judge_file).replace(".jsonl", "")

        print(f"\n{'='*70}")
        print(f"Judge Model: {model_name}")
        print("=" * 70)

        try:
            result = compute_expert_agreement(
                judge_file=judge_file,
                expert_csv=expert_csv,
                label_jsonl=label_jsonl,
                id_column=id_column,
                remove_evaluators=remove_evaluators,
                output_filename=None,  # Don't save individual files
            )
            results[model_name] = result
        except Exception as e:
            print(f"Error processing {judge_file}: {e}")
            results[model_name] = {"error": str(e)}

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: ALL JUDGE MODELS vs EXPERT")
    print("=" * 70)
    print(f"{'Model':<30} {'Pearson':>10} {'Spearman':>10} {'t-test p':>10}")
    print("-" * 70)

    for model_name, result in results.items():
        if "error" in result:
            print(f"{model_name:<30} {'ERROR':>10}")
        else:
            overall = result.get("overall_agreement", {})
            if "error" not in overall:
                # Get overall score metrics
                per_crit = result.get("per_criterion", {})
                overall_m = per_crit.get("Overall Score", per_crit.get("Overall Score AVG", {}))
                if overall_m and "error" not in overall_m:
                    print(f"{model_name:<30} {overall_m['pearson_r']:>10.2f} {overall_m['spearman_r']:>10.2f} {overall_m['ttest_p']:>10.2f}")
                else:
                    print(f"{model_name:<30} {overall['avg_pearson_r']:>10.2f} {overall['avg_spearman_r']:>10.2f} {'N/A':>10}")

    # Save combined results
    if output_filename is None:
        output_filename = "batch_expert_agreement.json"

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_filename}")

    return results


def self_agreement(
    prediction_file: str,
    model: str = "openai/gpt-4o-mini",
    n_runs: int = 2,
    prompt_name: str = "judge",
):
    """
    Measure self-agreement by running the same judge multiple times.

    This runs the judge n_runs times with temperature > 0 and measures
    agreement between runs.

    Args:
        prediction_file: Path to prediction JSONL file
        model: Judge model to use
        n_runs: Number of judge runs (default: 2)
        prompt_name: Prompt template name
    """
    import os
    from judge import RecipeJudge
    from tqdm import tqdm

    ds = list(jsonlines.open(prediction_file))
    prompt_filename = f"prompts/{prompt_name}.txt"

    # Use temperature > 0 for variability
    judge = RecipeJudge(
        model=model,
        prompt_filename=prompt_filename,
        temperature=0.3,
    )

    model_name = model.split("/")[-1]
    base_output = prediction_file.replace(".jsonl", f"_judged_{model_name}")

    output_files = []

    for run_idx in range(n_runs):
        output_filename = f"{base_output}_run{run_idx + 1}.jsonl"
        output_files.append(output_filename)

        # Check if already completed
        if os.path.exists(output_filename):
            existing = len(list(jsonlines.open(output_filename)))
            if existing >= len(ds):
                print(f"Run {run_idx + 1} already complete: {output_filename}")
                continue
            run_ds = ds[existing:]
            print(f"Run {run_idx + 1}: Resuming from {existing}/{len(ds)}")
        else:
            run_ds = ds
            print(f"Run {run_idx + 1}: Starting fresh")

        with jsonlines.open(output_filename, "a") as fout:
            for item in tqdm(run_ds, desc=f"Run {run_idx + 1}"):
                judge_result = judge.judge(item)
                item["judge_result"] = judge_result
                item["judge_model"] = model
                item["run_idx"] = run_idx + 1
                fout.write(item)

    # Compute agreement between runs
    if n_runs == 2:
        compute_agreement(output_files[0], output_files[1])
    else:
        # Pairwise agreement for multiple runs
        from itertools import combinations
        for f1, f2 in combinations(output_files, 2):
            print(f"\nComparing {os.path.basename(f1)} vs {os.path.basename(f2)}")
            compute_agreement(f1, f2)


if __name__ == "__main__":
    fire.Fire({
        "compare": compute_agreement,
        "expert": compute_expert_agreement,
        "batch_expert": batch_expert_agreement,
        "self": self_agreement,
    })
