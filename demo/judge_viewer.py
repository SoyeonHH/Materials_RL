"""
Streamlit demo for viewing judge results and expert evaluations.

Usage:
    streamlit run demo/app.py
"""

import json
import os
import re
import glob
import unicodedata

import pandas as pd
import streamlit as st
import jsonlines


# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "human_eval")
LABEL_FILE = os.path.join(DATA_DIR, "mixed-10test-0117.jsonl")


def normalize_path(path: str) -> str:
    """Normalize file path to handle NFD/NFC unicode differences."""
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    if os.path.exists(path):
        return path
    if directory and os.path.exists(directory):
        for f in os.listdir(directory):
            if unicodedata.normalize('NFC', f) == unicodedata.normalize('NFC', filename):
                return os.path.join(directory, f)
    return path


def find_expert_csv() -> str | None:
    """Find the expert evaluation CSV file."""
    # Handle unicode normalization issues in filenames
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            normalized = unicodedata.normalize('NFC', f)
            if normalized.endswith('.csv') and '예측평가' in normalized:
                return os.path.join(DATA_DIR, f)
    return None


def find_judge_files() -> dict[str, str]:
    """Find all judge result files and return dict of model_name -> filepath."""
    judge_files = {}
    for f in glob.glob(os.path.join(DATA_DIR, "mixed-10test-0117_*.jsonl")):
        basename = os.path.basename(f)
        # Extract model name from filename
        # Format: mixed-10test-0117_{model}_{prompt}.jsonl or mixed-10test-0117_judged_{model}.jsonl
        if "judged_" in basename:
            # Old format: mixed-10test-0117_judged_gpt-4o.jsonl
            match = re.search(r"_judged_(.+)\.jsonl", basename)
            if match:
                model_name = match.group(1)
                judge_files[f"{model_name} (judge)"] = f
        else:
            # New format: mixed-10test-0117_gpt-5-mini_judge_1shot.jsonl
            match = re.search(r"mixed-10test-0117_(.+?)_(judge.*?)\.jsonl", basename)
            if match:
                model_name = match.group(1)
                prompt_name = match.group(2)
                judge_files[f"{model_name} ({prompt_name})"] = f
    return judge_files


@st.cache_data
def load_instances() -> list[dict]:
    """Load instances from the label JSONL file."""
    instances = []
    with jsonlines.open(LABEL_FILE) as reader:
        for item in reader:
            instances.append(item)
    return instances


@st.cache_data
def load_expert_evaluations(csv_path: str) -> pd.DataFrame:
    """Load expert evaluation CSV."""
    csv_path = normalize_path(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    return df


def load_judge_results(filepath: str) -> dict[str, dict]:
    """Load judge results from JSONL file, indexed by combined id-model key."""
    results = {}
    with jsonlines.open(filepath) as reader:
        for item in reader:
            # Create combined key: paper-04-o1-mini
            sample_id = item.get("id", "")
            model = item.get("model", "")
            key = f"{sample_id}-{model}" if model else sample_id
            # Store with source file info for debugging
            item["_source_file"] = filepath
            results[key] = item
    return results


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


def get_instance_display_name(instance: dict) -> str:
    """Get display name for instance selector."""
    paper_id = instance.get("id", "unknown")
    model = instance.get("model", "")
    classification = instance.get("classification_result", "")

    # Extract material info from classification
    material_match = re.search(r"Material Name:\s*(.+?)(?:\n|$)", classification)
    material_name = material_match.group(1).strip() if material_match else ""

    if model:
        return f"{paper_id} | {model} | {material_name[:30]}"
    return f"{paper_id} | {material_name[:40]}"


# Score columns and their explanation column indices
# CSV format: Score1, Explanation, Score2, Explanation.1, ...
SCORE_COLUMNS_WITH_EXPLANATION = [
    ("Material_Appropriateness", "Explanation"),
    ("Material_Quantities", "Explanation.1"),
    ("Material_Specifications", "Explanation.2"),
    ("Equipment_Appropriateness", "Explanation.3"),
    ("Equipment_Specifications", "Explanation.4"),
    ("Procedure_Completeness", "Explanation.5"),
    ("Procedure_Concreteness", "Explanation.6"),
    ("Procedure_Similarity", "Explanation.7"),
    ("Procedure_Feasibility", "Explanation.8"),
    ("Characterization_Appropriateness", "Explanation.9"),
    ("Characterization_Similarity", "Explanation.10"),
    ("Overall Score", "Explanation.11"),
]

# Main score columns for summary
SCORE_COLUMNS = [
    "Material_Appropriateness",
    "Equipment_Appropriateness",
    "Procedure_Completeness",
    "Procedure_Similarity",
    "Procedure_Feasibility",
    "Characterization_Appropriateness",
    "Characterization_Similarity",
    "Overall Score",
]

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


def main():
    st.set_page_config(
        page_title="Materials Recipe Evaluation Viewer",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Materials Recipe Evaluation Viewer")
    st.markdown("View and compare model predictions, expert evaluations, and LLM judge results.")

    # Load data
    instances = load_instances()
    expert_csv = find_expert_csv()
    judge_files = find_judge_files()

    if not instances:
        st.error("No instances found in the label file.")
        return

    # Load expert data first (needed for instance selector)
    expert_df = None
    if expert_csv:
        expert_df = load_expert_evaluations(expert_csv)

    # Build instance options with expert scores
    def get_expert_overall_score(idx: int) -> float | None:
        """Get expert overall score for instance at given index."""
        if expert_df is None:
            return None
        row_id = f"Row-{idx:02d}"
        expert_rows = expert_df[expert_df["Paper ID"] == row_id]
        if expert_rows.empty or "Overall Score" not in expert_rows.columns:
            return None
        return expert_rows["Overall Score"].mean()

    # Create instance options with expert scores, sorted by score descending
    instance_data = []
    for idx, inst in enumerate(instances):
        expert_score = get_expert_overall_score(idx)
        display_name = get_instance_display_name(inst)
        if expert_score is not None:
            display_with_score = f"[{expert_score:.1f}] {display_name}"
        else:
            display_with_score = f"[N/A] {display_name}"
        instance_data.append((display_with_score, idx, expert_score if expert_score is not None else -1))

    # Sort by expert score descending (highest first)
    instance_data.sort(key=lambda x: x[2], reverse=True)

    # Sidebar: Instance and Judge selection
    st.sidebar.header("Selection")

    # Instance selector
    instance_options = {item[0]: item[1] for item in instance_data}
    selected_display = st.sidebar.selectbox(
        "Select Instance",
        options=list(instance_options.keys()),
        index=0,
    )
    selected_idx = instance_options[selected_display]
    instance = instances[selected_idx]

    # Judge model selector
    selected_judge = None
    if judge_files:
        selected_judge = st.sidebar.selectbox(
            "Select Judge Model",
            options=list(judge_files.keys()),
            index=0,
        )

    # Load judge results if selected
    judge_results = None
    if selected_judge:
        judge_results = load_judge_results(judge_files[selected_judge])

    # Get instance identifiers
    paper_id = instance.get("id", "")
    model_name = instance.get("model", "")
    row_id = f"Row-{selected_idx:02d}"
    combined_id = f"{paper_id}-{model_name}" if model_name else paper_id

    # === Instance Information Header ===
    st.header("📋 Instance Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Paper ID", paper_id)
    with col2:
        st.metric("Model", model_name if model_name else "N/A")
    with col3:
        st.metric("Row ID", row_id)

    st.divider()

    # === Main Content: 3 columns layout ===
    left_col, mid_col, right_col = st.columns([1, 1, 1])

    # Left Column: Input & Ground Truth
    with left_col:
        st.subheader("📝 Input (Classification)")
        st.text_area(
            "Classification",
            value=instance.get("classification_result", ""),
            height=250,
            disabled=True,
            key=f"classification_{combined_id}",
            label_visibility="collapsed",
        )

        st.subheader("✅ Ground Truth Recipe")
        st.text_area(
            "Recipe",
            value=instance.get("recipe", ""),
            height=400,
            disabled=True,
            key=f"recipe_{combined_id}",
            label_visibility="collapsed",
        )

    # Middle Column: Model Prediction
    with mid_col:
        st.subheader("🤖 Model Prediction")
        st.text_area(
            "Prediction",
            value=instance.get("prediction", ""),
            height=680,
            disabled=True,
            key=f"prediction_{combined_id}",
            label_visibility="collapsed",
        )

    # Right Column: Evaluations (Expert + Judge)
    with right_col:
        # Expert Evaluation Section
        st.subheader("👨‍🔬 Expert Evaluation")

        if expert_df is None:
            st.warning("Expert CSV not found.")
        else:
            expert_rows = expert_df[expert_df["Paper ID"] == row_id]

            if expert_rows.empty:
                st.info(f"No expert evaluations for {row_id}")
            else:
                # Calculate average scores
                score_cols_present = [c for c in SCORE_COLUMNS if c in expert_rows.columns]
                if score_cols_present:
                    avg_scores = expert_rows[score_cols_present].mean()

                    # Compact score display
                    score_df = pd.DataFrame({
                        "Criterion": [c.replace("_", " ") for c in score_cols_present],
                        "Avg Score": [f"{avg_scores[c]:.1f}" for c in score_cols_present],
                    })
                    st.dataframe(score_df, hide_index=True, use_container_width=True)

                # Individual evaluator details
                for _, row in expert_rows.iterrows():
                    evaluator = row.get("Evaluation", "Unknown")
                    group = row.get("Group", "")

                    with st.expander(f"📋 {evaluator} ({group})", expanded=False):
                        for score_col, expl_col in SCORE_COLUMNS_WITH_EXPLANATION:
                            if score_col in row.index:
                                score_val = row.get(score_col, "N/A")
                                expl_val = row.get(expl_col, "")
                                display_name = score_col.replace("_", " ")

                                score_str = str(int(score_val)) if pd.notna(score_val) and isinstance(score_val, float) else str(score_val)
                                st.write(f"**{display_name}**: {score_str}")
                                if pd.notna(expl_val) and str(expl_val).strip():
                                    st.caption(f"_{expl_val}_")

        st.divider()

        # Judge Evaluation Section
        st.subheader("🤖 Judge Evaluation")

        if not selected_judge:
            st.warning("No judge results available.")
        elif judge_results is None:
            st.warning("Failed to load judge results.")
        else:
            judge_item = judge_results.get(combined_id)
            if judge_item is None:
                judge_item = judge_results.get(paper_id)

            if judge_item is None:
                st.info(f"No judge result for {combined_id}")
            else:
                st.caption(f"**Judge:** {selected_judge}")

                # Debug: show lookup info
                judge_item_id = judge_item.get("id", "")
                judge_item_model = judge_item.get("model", "")
                source_file = judge_item.get("_source_file", "unknown")
                st.caption(f"_Looking for: {combined_id} | Found: {judge_item_id}-{judge_item_model}_")
                st.caption(f"_Source: {os.path.basename(source_file)}_")

                judge_result_text = judge_item.get("judge_result", "")
                extracted_scores = extract_json_from_text(judge_result_text)

                if extracted_scores:
                    # Compact score display
                    judge_score_data = []
                    for key in JUDGE_SCORE_KEYS:
                        if key in extracted_scores:
                            display_name = key.replace("_score", "").replace("_", " ").title()
                            score_val = extracted_scores[key]
                            # Format to 2 decimal places to preserve original precision
                            score_str = f"{float(score_val):.2f}" if isinstance(score_val, (int, float)) else str(score_val)
                            judge_score_data.append({
                                "Criterion": display_name,
                                "Score": score_str,
                            })
                    if judge_score_data:
                        st.dataframe(pd.DataFrame(judge_score_data), hide_index=True, use_container_width=True)

                    # Show reasoning
                    with st.expander("💭 Reasoning", expanded=False):
                        for key, value in extracted_scores.items():
                            if "reason" in key.lower() or "explanation" in key.lower():
                                st.write(f"**{key}:**")
                                st.write(value)

                # Full response
                with st.expander("📝 Full Judge Response", expanded=False):
                    st.text_area(
                        "Judge Result",
                        value=judge_result_text,
                        height=400,
                        disabled=True,
                        key=f"judge_result_{combined_id}_{selected_judge}",
                        label_visibility="collapsed",
                    )

    # Sidebar: Data stats
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Data Statistics")
    st.sidebar.write(f"Total Instances: {len(instances)}")
    st.sidebar.write(f"Judge Models: {len(judge_files)}")
    if expert_df is not None:
        st.sidebar.write(f"Expert Evaluations: {len(expert_df)}")


if __name__ == "__main__":
    main()
