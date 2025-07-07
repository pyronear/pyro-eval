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
available_runs.insert(0, default_value)
with col1:
    st.markdown("### Reference Run")
    run_a = st.selectbox("Reference Run", available_runs, key="run_a")
with col2:
    st.markdown("### Compared to ...")
    run_b = st.selectbox("Compared to ...", available_runs, key="run_b")

runs = []

if run_a != run_b and run_a != default_value and run_b != default_value:
    try:
        runs.append(RunData(run_a))
        runs.append(RunData(run_b))
    except Exception as e:
        st.error(f"Error loading runs : {e}\nResults should be stored in metrics.json")
        st.subheader("Expected predictions format in metrics.json")
        st.code("""
{
"predictions": {
    "tp": ["image1.jpg", "image2.jpg", ...],
    "fp": ["image3.jpg", "image4.jpg", ...],
    "fn": ["image5.jpg", "image6.jpg", ...],
    "tn": ["image7.jpg", "image8.jpg", ...]
}
}
        """, language="json")

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
        changed_count = len(df[df["Change Type"] != "unchanged"])
        st.metric(
            "Status changed",
            changed_count,
            delta=None
        )
    
    with col3:
        improved_count = len(df[df["Change Type"] == 'improved'])
        st.metric(
            "Improved",
            improved_count,
            delta=f"{improved_count/len(df)*100:.1f}%"
        )
    
    with col4:
        degraded_count = len(df[df['Change Type'] == 'degraded'])
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
    
    change_filter = st.sidebar.selectbox(
                "Change type",
                options=['All', 'Improvements', 'Degradations', 'Unchanged', 
                        'FP → TP', 'FN → TP', 'TP → FP', 'TP → FN', 'Other changes']
            )
            
    # Filter by researching an image
    search_term = st.sidebar.text_input("Recherche d'image", "")
    
    # Apply filters
    filtered_df = df.copy()
    
    if change_filter == 'Improvements':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'improved']
    elif change_filter == 'Degradations':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'degraded']
    elif change_filter == 'Unchanged':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'unchanged']
    elif change_filter == 'FP → TP':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'fp-to-tn']
    elif change_filter == 'FN → TP':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'fn-to-tp']
    elif change_filter == 'TP → FP':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'tp-to-fn']
    elif change_filter == 'TP → FN':
        filtered_df = filtered_df[filtered_df['Change Type'] == 'tn-to-fp']
    elif change_filter == 'Other changes':
        filtered_df = filtered_df[~filtered_df['Change Type'].isin(['fp-to-tn', 'fn-to-tp', 'tn-to-fp', 'tp-to-fn', 'unchanged'])]
    
    if search_term:
        filtered_df = filtered_df[filtered_df['Image Name'].str.contains(search_term, case=False)]

    st.header(f"Filtered comparison : ({len(filtered_df)} images)")
            
    if not filtered_df.empty:
        display_df = filtered_df.copy()
        
        display_df[run_a] = display_df[run_a].apply(
            lambda x: comparison.display_status_badge(x)
        )
        display_df[run_b] = display_df[run_b].apply(
            lambda x: comparison.display_status_badge(x)
        )
        
        # Add a colonne to indicate change
        display_df['Indicator'] = display_df.apply(
            lambda row: "🔄" if row["Change Type"] != "unchanged" else "➖", axis=1
        )
        
        display_df = display_df[['Image Name', run_a, 'Indicator', run_b, 'Transition']]
        
        # HTML table for color badges
        st.markdown(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        
        # CSV export
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="comparison_results.csv",
            mime="text/csv"
        )

    else:
        st.info("Load two runs to start the comparison")
