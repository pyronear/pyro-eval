import streamlit as st
from pathlib import Path

import pandas as pd
from run_data import RunData, RunComparison, PROJECT_ROOT

st.set_page_config(
    page_title="Comparaison de Modèles de Détection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status-tp { background-color: #d4edda; color: #155724; }
    .status-fp { background-color: #f8d7da; color: #721c24; }
    .status-fn { background-color: #fff3cd; color: #856404; }
    .status-tn { background-color: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>Run comparison</h1>
        <p>Compare metrics and predictions on Evaluation Pipeline results</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.header("📁 Configuration")

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
    status = comparison.compare_predictions(source="model")
elif compare_engine:
    status = comparison.compare_predictions(source="sequence")
else:
    status = None

# # --- Format DataFrame pour affichage ---
if status:
    # df = pd.DataFrame.from_dict(status, orient="index")
    df = comparison.get_status_dataframe(status=status)
    df.index.name = "image"

    st.header("Global statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total number of images/sequences",
            len(df),
            delta=None
        )
    
    with col2:
        changed_count = len(df[df["change_type"] != "unchanged"])
        st.metric(
            "Status changed",
            changed_count,
            delta=None
        )
    
    with col3:
        improved_count = len(df[df["change_type"] == 'improved'])
        st.metric(
            "Improved",
            improved_count,
            delta=f"{improved_count/len(df)*100:.1f}%"
        )
    
    with col4:
        degraded_count = len(df[df['change_type'] == 'degraded'])
        st.metric(
            "Degraded",
            degraded_count,
            delta=f"-{degraded_count/len(df)*100:.1f}%"
        )
    
    st.header("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            comparison.create_confusion_matrix(df),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            comparison.create_change_distribution(df),
            use_container_width=True
        )
    
    st.dataframe(df)