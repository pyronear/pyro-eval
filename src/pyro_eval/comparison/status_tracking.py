import streamlit as st
from pathlib import Path
from typing import List

import pandas as pd

from image_manager import ImageManager
from run_data import RunData, RunComparison, PROJECT_ROOT
from utils import compare_metrics

st.set_page_config(
    page_title="Wildfire Detection Model Comparison",
    page_icon="🌲",
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

# @st.cache_data
def create_image_manager(
    runs : List[RunData]
) -> ImageManager:
    return ImageManager(runs=runs)

# If we want to use streamlit cache for create_image_manager, we need to have hashable input to the function
# For that we need to provide a dict version of the run objects...
# @st.cache_data
# def create_image_manager(runs_data: List[dict]) -> ImageManager:
#     runs = [RunData(**data) for data in runs_data]
#     return ImageManager(runs=runs)

# ... and implement a to_dict() method in RunData
#create_image_manager([run.to_dict() for run in runs])


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
    d.name for d in eval_dir.iterdir()
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
    st.markdown("### New Run")
    run_b = st.selectbox("Compared to ...", available_runs, key="run_b")

runs = []

if run_a != run_b and run_a != default_value and run_b != default_value:
    try:
        runs.append(RunData(run_a))
        runs.append(RunData(run_b))
        image_manager = create_image_manager(runs)
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


# Store in session_state to avoid empty re-runs
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = None
if compare_model:
    st.session_state.compare_mode = "model"
if compare_engine:
    st.session_state.compare_mode = "engine"

if st.session_state.compare_mode:
    status = comparison.compare_predictions(source=st.session_state.compare_mode)
else:
    status = None

if status:

    # ==================================
    # === Global Comparison Display ====
    # ==================================
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

    # ==============================================
    # === Configuration Comparison (Collapsible) ===
    # ==============================================

    st.header("⚙️ Configuration Comparison")

    with st.expander("Show/Hide Configuration Comparison", expanded=False):
        
        st.session_state["config_expanded"] = True
        if len(runs) == 2:
            run_a_config = runs[0].config if hasattr(runs[0], 'config') else {}
            run_b_config = runs[1].config if hasattr(runs[1], 'config') else {}

            # Create comparison DataFrame for configs
            config_comparison = []

            # Get all unique keys from both configs
            all_keys = set()
            def extract_keys(d, prefix=""):
                for k, v in d.items():
                    if isinstance(v, dict):
                        extract_keys(v, prefix + k + ".")
                    else:
                        all_keys.add(prefix + k)
            
            extract_keys(run_a_config)
            extract_keys(run_b_config)
            
            # Function to get nested value
            def get_nested_value(d, key):
                keys = key.split('.')
                value = d
                try:
                    for k in keys:
                        value = value[k]
                    return value
                except (KeyError, TypeError):
                    return "N/A"
            
            # Compare configurations
            for key in sorted(all_keys):
                value_a = get_nested_value(run_a_config, key)
                value_b = get_nested_value(run_b_config, key)
                
                # Determine if values are different
                is_different = str(value_a) != str(value_b)
                
                config_comparison.append({
                    'Parameter': key,
                    run_a: str(value_a),
                    run_b: str(value_b),
                })
            
            config_df = pd.DataFrame(config_comparison)
            
            # Display config comparison with highlighting
            st.subheader("Configuration Parameters")
            show_only_different = st.checkbox("Show only different parameters", key="show_different_config")
            
            if show_only_different:
                config_df_filtered = config_df[config_df[run_a] != config_df[run_b]]
            else:
                config_df_filtered = config_df
            
            if not config_df_filtered.empty:
                st.dataframe(config_df_filtered, use_container_width=True)
            else:
                st.info("No configuration data available")
        else:
            st.warning("Need at least 2 runs to compare configurations")


    # ==============================================
    # ====== Metrics Comparison (Collapsible) ======
    # ==============================================

    st.header("📊 Metrics Comparison")

    with st.expander("Show/Hide Metrics Comparison", expanded=False):
        
        st.session_state["metrics_expanded"] = True
        if len(runs) == 2:
            run_a_engine_metrics = runs[0].engine_metrics.get("sequence_metrics")
            run_a_engine_metrics.pop("predictions")
            run_b_engine_metrics = runs[1].engine_metrics.get("sequence_metrics")
            run_b_engine_metrics.pop("predictions")

            run_a_model_metrics = runs[0].model_metrics
            run_a_model_metrics.pop("predictions")
            run_a_model_metrics.pop("roc_curve")
            run_b_model_metrics = runs[1].model_metrics
            run_b_model_metrics.pop("predictions")
            run_b_model_metrics.pop("roc_curve")
            # Define preferred order for metrics
            preferred_order = ['f1', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn','avg_detection_delay',]
            
            engine_metrics_df = compare_metrics(run_a_engine_metrics, run_b_engine_metrics, preferred_order, [run_a, run_b])
            # Display metrics comparison with highlighting
            st.subheader("Engine Metrics")
            
            if not engine_metrics_df.empty:
                st.dataframe(engine_metrics_df, use_container_width=True)
            else:
                st.info("No metrics data available")
            
            model_metrics_df = compare_metrics(run_a_model_metrics, run_b_model_metrics, preferred_order, [run_a, run_b])
            # Display metrics comparison with highlighting
            st.subheader("Model Metrics")
            
            if not model_metrics_df.empty:
                st.dataframe(model_metrics_df, use_container_width=True)
            else:
                st.info("No metrics data available")
            
        else:
            st.warning("Need at least 2 runs to compare metrics")


    # ===============================================
    # ============ Dataframe and filters ============
    # ===============================================

    st.header("🔍 Prediction details")

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

    with col1:
        change_types = ['All'] + sorted(df['Change Type'].unique().tolist())
        change_filter = st.selectbox("Change Type", options=change_types, key="change_type_filter")

    with col2:
        run_a_vals = ['All'] + sorted(df[run_a].unique().tolist())
        run_a_filter = st.selectbox(f"{run_a} Status", options=run_a_vals, key="run_a_filter")

    with col3:
        run_b_vals = ['All'] + sorted(df[run_b].unique().tolist())
        run_b_filter = st.selectbox(f"{run_b} Status", options=run_b_vals, key="run_b_filter")

    with col4:
        search_term = st.text_input("Search Image", "", key="image_search")

    with col5:
        filter_mode = st.radio("Filter Mode", ["AND", "OR"], key="filter_mode")

    def apply_filters(df, filters, mode="AND"):
        """
        Apply filters with specified mode AND/OR
        """
        if mode == "AND":
            # Add filters on top of each others
            filtered_df = df.copy()
            for condition in filters:
                if condition is not None:
                    filtered_df = filtered_df[condition(filtered_df)]
            return filtered_df
        
        else:  # mode == "OR"
            valid_filters = [f for f in filters if f is not None]
            if not valid_filters:
                return df.copy()
            
            # Create an OR condition combined
            combined_condition = None
            for condition in valid_filters:
                if combined_condition is None:
                    combined_condition = condition(df)
                else:
                    combined_condition = combined_condition | condition(df)
            
            return df[combined_condition]


    # Create filter conditions for each select_box
    filter_conditions = []

    if change_filter != 'All':
        filter_conditions.append(lambda x: x['Change Type'] == change_filter)
    else:
        filter_conditions.append(None)

    if run_a_filter != 'All':
        filter_conditions.append(lambda x: x[run_a] == run_a_filter)
    else:
        filter_conditions.append(None)

    if run_b_filter != 'All':
        filter_conditions.append(lambda x: x[run_b] == run_b_filter)
    else:
        filter_conditions.append(None)

    if search_term:
        filter_conditions.append(lambda x: x['Name'].str.contains(search_term, case=False))
    else:
        filter_conditions.append(None)

    filtered_df = apply_filters(df, filter_conditions, filter_mode)

    # Display
    st.data_editor(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        disabled=True  # désactive l’édition si ce n’est pas souhaité
    )

    # Display filter summary
    st.info(f"Showing {len(filtered_df)} of {len(df)} images/sequences")
    
    # Display filtered results
    if not filtered_df.empty:
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export filtered results
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Results (CSV)",
                data=csv_filtered,
                file_name=f"filtered_comparison_{len(filtered_df)}_images.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export full results
            csv_full = df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv_full,
                file_name=f"full_comparison_{len(df)}_images.csv",
                mime="text/csv"
            )

        with col3:
            # Export full results
            if st.button("Export filtered image folder"):
                image_manager.create_image_folder(
                    df=filtered_df,
                    out_path="/Users/theocayla/Documents/Dev/Pyronear/debug/interface/data/out",
                    source="engine",
                    query="query_example"
                )

        # Additional statistics for filtered results
        if len(filtered_df) != len(df):
            st.subheader("📊 Filtered Statistics")
            
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                st.metric("Filtered Entries", len(filtered_df))
            
            with filter_col2:
                filtered_changed = len(filtered_df[filtered_df["Change Type"] != "unchanged"])
                st.metric("Changed (Filtered)", filtered_changed)
            
            with filter_col3:
                filtered_degraded = len(filtered_df[filtered_df["Change Type"] == 'improved'])
                st.metric("Improved (Filtered)", filtered_degraded)
            
            with filter_col4:
                filtered_degraded = len(filtered_df[filtered_df['Change Type'] == 'degraded'])
                st.metric("Degraded (Filtered)", filtered_degraded)
    
    else:
        st.warning("No images match the current filters. Try adjusting your filter criteria.")
        
        # Show current filter state
        st.subheader("Current Filters:")
        st.write(f"- Change Type: {change_filter}")
        st.write(f"- {run_a} Status: {run_a_filter}")
        st.write(f"- {run_b} Status: {run_b_filter}")
        if search_term:
            st.write(f"- Search Term: '{search_term}'")

    # ================================
    # ============ Graphs ============
    # ================================

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