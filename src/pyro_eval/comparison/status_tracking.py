import streamlit as st
from pathlib import Path

import pandas as pd
from run_data import RunData, RunComparison, PROJECT_ROOT

st.title("Compare model outputs")

# Get avaialable runs
eval_dir = PROJECT_ROOT / "data/evaluation"
available_runs = sorted([
    d.name for d in Path("data/evaluation").iterdir()
    if d.is_dir() and (d / "metrics.json").is_file()
])

# Select runs
col1, col2 = st.columns(2)
default_value = "Select a run"
available_runs.insert(0, "Select a run")
with col1:
    run_a = st.selectbox("Run A", available_runs, key="run_a")
with col2:
    run_b = st.selectbox("Run B", available_runs, key="run_b")

runs = []

# Charger les runs sélectionnés
if run_a != run_b and run_a != default_value and run_b != default_value:
    try:
        runs.append(RunData(run_a))
        runs.append(RunData(run_b))
    except Exception as e:
        st.error(f"Error loading runs : {e}")

# --- Comparaison ---
comparison = RunComparison(runs=runs)
col1, col2 = st.columns(2)
with col1:
    compare_model = st.button("Compare model predictions")
with col2:
    compare_engine = st.button("Compare engine predictions")

if compare_model:
    diff = comparison.get_changed_status(source="model")
elif compare_engine:
    diff = comparison.get_changed_status(source="sequence")
else:
    diff = None

# # --- Format DataFrame pour affichage ---
if diff:
    df = pd.DataFrame.from_dict(diff, orient="index")
    df.index.name = "image"
    st.dataframe(df)
