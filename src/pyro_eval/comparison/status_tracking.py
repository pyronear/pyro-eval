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
    st.markdown("### New Run")
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


# Store in session_state to avoid empty re-runs
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = None
if compare_model:
    st.session_state.compare_mode = "model"
if compare_engine:
    st.session_state.compare_mode = "sequence"

if st.session_state.compare_mode:
    status = comparison.compare_predictions(source=st.session_state.compare_mode)
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
    
# --- Configuration Comparison (Collapsible) ---
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
    
    # # --- Dataframe and filters ---
    st.header("🔍 Prediciton details")

    # Implement filters
    col1, col2, col3, col4 = st.columns(4)

    # Filter given some specific changes
    with col1:
        change_types = ['All'] + sorted(df['Change Type'].unique().tolist())
        change_filter = st.selectbox("Change Type", options=change_types, key="change_type_filter")

    # Filter on run_a values
    with col2:
        run_a_vals = ['All'] + sorted(df[run_a].unique().tolist())
        run_a_filter = st.selectbox(f"{run_a} Status", options=run_a_vals, key="run_a_filter")

    # Filter on run_b values
    with col3:
        run_b_vals = ['All'] + sorted(df[run_b].unique().tolist())
        run_b_filter = st.selectbox(f"{run_b} Status", options=run_b_vals, key="run_b_filter")

    with col4:
        search_term = st.text_input("Search Image", "", key="image_search")

    # Apply filters
    filtered_df = df.copy()

    if change_filter != 'All':
        filtered_df = filtered_df[filtered_df['Change Type'] == change_filter]
    if run_a_filter != 'All':
        filtered_df = filtered_df[filtered_df[run_a] == run_a_filter]
    if run_b_filter != 'All':
        filtered_df = filtered_df[filtered_df[run_b] == run_b_filter]
    if search_term:
        filtered_df = filtered_df[filtered_df['Image Name'].str.contains(search_term, case=False)]

    # Display
    # st.data_editor(
    #     filtered_df,
    #     use_container_width=True,
    #     hide_index=True,
    #     num_rows="dynamic",
    #     disabled=True  # désactive l’édition si ce n’est pas souhaité
    # )

    # Display filter summary
    st.info(f"Showing {len(filtered_df)} of {len(df)} images/sequences")
    
    # Display filtered results
    if not filtered_df.empty:
        display_df = filtered_df.copy()
        
        # Apply status badges
        display_df[run_a] = display_df[run_a].apply(
            lambda x: comparison.display_status_badge(x)
        )
        display_df[run_b] = display_df[run_b].apply(
            lambda x: comparison.display_status_badge(x)
        )
        
        # Add indicator column
        display_df['Indicator'] = display_df.apply(
            lambda row: "🔄" if row["Change Type"] != "unchanged" else "➖", axis=1
        )
        
        # Reorder columns for display
        display_df = display_df[['Image Name', run_a, 'Indicator', run_b]]
        
        # Display table with HTML for badges
        st.markdown(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        
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
        

        # Additional statistics for filtered results
        if len(filtered_df) != len(df):
            st.subheader("📊 Filtered Statistics")
            
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                st.metric("Filtered Images", len(filtered_df))
            
            with filter_col2:
                filtered_changed = len(filtered_df[filtered_df["Change Type"] != "unchanged"])
                st.metric("Changed (Filtered)", filtered_changed)
            
            with filter_col3:
                filtered_degraded = len(filtered_df[filtered_df["Change Type"] == 'degraded'])
                st.metric("Degraded (Filtered)", filtered_degraded)
            
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

# ------- Graphs -------

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