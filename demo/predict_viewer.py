"""
Streamlit demo for viewing prediction results and evaluation scores across models.

Usage:
    streamlit run demo/predict_viewer.py
"""

import json
import os
import re
import glob

import pandas as pd
import streamlit as st
import jsonlines


# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "test_high_impact")

# Score keys for judge results
JUDGE_SCORE_KEYS = [
    "materials_appropriateness_score",
    "equipment_appropriateness_score",
    "procedure_completeness_score",
    "procedure_similarity_score",
    "procedure_feasibility_score",
    "characterization_appropriateness_score",
    "characterization_similarity_score",
    "overall_score",
]


def find_model_folders() -> list[str]:
    """Find all model folders in the data directory."""
    if not os.path.exists(DATA_DIR):
        return []
    return sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])


def find_prediction_file(model_folder: str) -> str | None:
    """Find the prediction.jsonl file for a model."""
    pred_path = os.path.join(DATA_DIR, model_folder, "prediction.jsonl")
    if os.path.exists(pred_path):
        return pred_path
    return None


def find_judged_files(model_folder: str) -> dict[str, str]:
    """Find all judged result files for a model."""
    judged_files = {}
    pattern = os.path.join(DATA_DIR, model_folder, "prediction_*_judge*.jsonl")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        # Extract judge model and prompt from filename
        # Format: prediction_{judge_model}_{prompt}.jsonl
        match = re.search(r"prediction_(.+?)_(judge.*?)\.jsonl", basename)
        if match:
            judge_model = match.group(1)
            prompt_name = match.group(2)
            judged_files[f"{judge_model} ({prompt_name})"] = f
    return judged_files


def find_score_files(model_folder: str) -> dict[str, str]:
    """Find all score JSON files for a model."""
    score_files = {}
    pattern = os.path.join(DATA_DIR, model_folder, "*_scores.json")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        score_files[basename] = f
    return score_files


@st.cache_data
def load_predictions(filepath: str) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with jsonlines.open(filepath) as reader:
        for item in reader:
            predictions.append(item)
    return predictions


@st.cache_data
def load_judged_results(filepath: str) -> dict[str, dict]:
    """Load judged results from JSONL file, indexed by id."""
    results = {}
    with jsonlines.open(filepath) as reader:
        for item in reader:
            sample_id = item.get("id", "")
            results[sample_id] = item
    return results


@st.cache_data
def load_scores(filepath: str) -> dict:
    """Load aggregated scores from JSON file."""
    with open(filepath) as f:
        return json.load(f)


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


def get_instance_display_name(instance: dict, idx: int) -> str:
    """Get display name for instance selector."""
    sample_id = instance.get("id", "unknown")[:8]
    contribution = instance.get("contribution", "")

    # Extract material info from contribution
    material_match = re.search(r"Novel materials or compounds:\s*(.+?)(?:\n|$)", contribution)
    material_name = material_match.group(1).strip()[:40] if material_match else ""

    return f"{idx+1:03d} | {sample_id} | {material_name}"


def main():
    st.set_page_config(
        page_title="Materials Recipe Prediction Viewer",
        page_icon="🧪",
        layout="wide",
    )

    st.title("🧪 Materials Recipe Prediction Viewer")
    st.markdown("Compare model predictions and evaluation scores across different models.")

    # Find available models
    model_folders = find_model_folders()

    if not model_folders:
        st.error(f"No model folders found in {DATA_DIR}")
        return

    # Sidebar: Model and Instance selection
    st.sidebar.header("Selection")

    # Model selector
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_folders,
        index=0,
    )

    # Load prediction file
    pred_file = find_prediction_file(selected_model)
    if not pred_file:
        st.error(f"No prediction.jsonl found for {selected_model}")
        return

    predictions = load_predictions(pred_file)

    if not predictions:
        st.error("No predictions found in the file.")
        return

    # Find judged files for this model
    judged_files = find_judged_files(selected_model)
    score_files = find_score_files(selected_model)

    # Judge selector (if available)
    selected_judge = None
    judged_results = None
    if judged_files:
        selected_judge = st.sidebar.selectbox(
            "Select Judge",
            options=list(judged_files.keys()),
            index=0,
        )
        judged_results = load_judged_results(judged_files[selected_judge])

    # Instance selector with scores
    def get_instance_score(inst: dict) -> float:
        """Get overall score for an instance if judged."""
        if judged_results is None:
            return -1
        judged_item = judged_results.get(inst.get("id", ""))
        if judged_item is None:
            return -1
        judge_result = judged_item.get("judge_result", "")
        extracted = extract_json_from_text(judge_result)
        if extracted and "overall_score" in extracted:
            return float(extracted["overall_score"])
        return -1

    # Build instance options with scores
    instance_data = []
    for idx, inst in enumerate(predictions):
        score = get_instance_score(inst)
        display_name = get_instance_display_name(inst, idx)
        if score >= 0:
            display_with_score = f"[{score:.1f}] {display_name}"
        else:
            display_with_score = f"[N/A] {display_name}"
        instance_data.append((display_with_score, idx, score))

    # Sort option
    sort_option = st.sidebar.radio(
        "Sort by",
        options=["Index", "Score (High to Low)", "Score (Low to High)"],
        index=0,
    )

    if sort_option == "Score (High to Low)":
        instance_data.sort(key=lambda x: x[2], reverse=True)
    elif sort_option == "Score (Low to High)":
        instance_data.sort(key=lambda x: (x[2] < 0, x[2]))  # Put N/A at end

    # Instance selector
    instance_options = {item[0]: item[1] for item in instance_data}
    selected_display = st.sidebar.selectbox(
        "Select Instance",
        options=list(instance_options.keys()),
        index=0,
    )
    selected_idx = instance_options[selected_display]
    instance = predictions[selected_idx]

    # Get identifiers
    sample_id = instance.get("id", "unknown")

    # === Model Stats in Sidebar ===
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Statistics")
    st.sidebar.write(f"**Model:** {selected_model}")
    st.sidebar.write(f"**Total Samples:** {len(predictions)}")

    # Show aggregated scores if available
    if score_files:
        st.sidebar.markdown("### Average Scores")
        for score_name, score_path in score_files.items():
            scores = load_scores(score_path)
            avg_scores = scores.get("average_scores", {})
            if avg_scores:
                overall = avg_scores.get("overall_score", "N/A")
                st.sidebar.metric("Overall Score", f"{overall:.2f}" if isinstance(overall, float) else overall)

                with st.sidebar.expander("All Criteria"):
                    for key, val in avg_scores.items():
                        if val is not None:
                            short_key = key.replace("_score", "").replace("_", " ").title()
                            st.write(f"{short_key}: {val:.2f}")

    # === Instance Information Header ===
    st.header("📋 Instance Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample ID", sample_id[:16] + "..." if len(sample_id) > 16 else sample_id)
    with col2:
        st.metric("Model", selected_model)
    with col3:
        st.metric("Instance", f"{selected_idx + 1} / {len(predictions)}")

    st.divider()

    # === Main Content: 3 columns layout ===
    left_col, mid_col, right_col = st.columns([1, 1, 1])

    # Left Column: Input (Contribution)
    with left_col:
        st.subheader("📝 Input (Contribution)")
        st.text_area(
            "Contribution",
            value=instance.get("contribution", ""),
            height=300,
            disabled=True,
            key=f"contribution_{sample_id}",
            label_visibility="collapsed",
        )

        st.subheader("✅ Ground Truth Recipe")
        st.text_area(
            "Ground Truth",
            value=instance.get("recipe", ""),
            height=400,
            disabled=True,
            key=f"recipe_{sample_id}",
            label_visibility="collapsed",
        )

    # Middle Column: Model Prediction
    with mid_col:
        st.subheader("🤖 Model Prediction")
        st.text_area(
            "Prediction",
            value=instance.get("prediction", ""),
            height=730,
            disabled=True,
            key=f"prediction_{sample_id}",
            label_visibility="collapsed",
        )

    # Right Column: Judge Evaluation
    with right_col:
        st.subheader("⚖️ Judge Evaluation")

        if not selected_judge or judged_results is None:
            st.info("No judge results available for this model.")
        else:
            judged_item = judged_results.get(sample_id)

            if judged_item is None:
                st.warning(f"No judge result for sample {sample_id}")
            else:
                st.caption(f"**Judge:** {selected_judge}")

                judge_result_text = judged_item.get("judge_result", "")

                # Full response only
                st.text_area(
                    "Judge Result",
                    value=judge_result_text,
                    height=680,
                    disabled=True,
                    key=f"judge_result_{sample_id}_{selected_judge}",
                    label_visibility="collapsed",
                )

    # === Model Comparison Section ===
    st.divider()
    st.header("📊 Model Comparison")

    # Collect scores from all models for this sample
    comparison_data = []
    for model in model_folders:
        jf = find_judged_files(model)
        if not jf:
            continue

        # Use first available judge file
        judge_name, judge_path = list(jf.items())[0]
        try:
            jr = load_judged_results(judge_path)
            item = jr.get(sample_id)
            if item:
                extracted = extract_json_from_text(item.get("judge_result", ""))
                if extracted:
                    row = {"Model": model}
                    for key in JUDGE_SCORE_KEYS:
                        if key in extracted:
                            short_key = key.replace("_score", "").replace("_", " ").title()
                            row[short_key] = float(extracted[key])
                    comparison_data.append(row)
        except Exception:
            continue

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No comparison data available for this sample.")


if __name__ == "__main__":
    main()
