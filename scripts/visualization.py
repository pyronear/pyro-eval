import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from .compare_metrics import build_dataframe


class App:
    def __init__(_self, run_dirs: List):
        _self.run_dirs = run_dirs
        _self.df = _self.load_dataframe()
        _self.filtered_df = _self.df.copy()
        _self.predictions = _self.load_predictions()
        _self.model_cols = [
            "run_id",
            "model_path",
            "model_precision",
            "model_recall",
            "model_f1",
            "model_fp",
            "model_tp",
            "model_fn",
        ]
        _self.engine_cols = [
            "run_id",
            "seq_precision",
            "seq_recall",
            "seq_f1",
            "seq_fp",
            "seq_tp",
            "seq_fn",
            "img_precision",
            "img_recall",
            "img_f1",
            "img_fp",
            "img_tp",
            "img_fn",
            "model_path",
            "conf_thresh",
            "nb_consecutive_frames",
            "iou",
            "max_bbox_size",
        ]
        _self.split_df = {
            "model": _self.df[_self.model_cols].copy(),
            "engine": _self.df[_self.engine_cols].copy(),
        }

    @st.cache_data
    def load_predictions(_self):
        """
        Loads detailed predictions from metrics.json in run directory
        """
        predictions = {}
        for run_dir in _self.run_dirs:
            pred_path = Path(run_dir) / "metrics.json"
            if not pred_path.exists():
                raise FileExistsError(pred_path)
            with open(pred_path, "r") as f:
                data = json.load(f)
            predictions[data["run_id"]] = {
                "model_preds": data.get("model_metrics", {}).get("predictions"),
                "engine_preds": data.get("engine_metrics", {})
                .get("sequence_metrics", {})
                .get("predictions"),
            }
        return predictions

    @st.cache_data
    def load_dataframe(_self):
        return build_dataframe(_self.run_dirs)

    def compare_predictions(_self, runs):
        st.subheader("Prediction Differences")
        pred1 = _self.predictions[runs[0]]
        pred2 = _self.predictions[runs[1]]

        if pred1 and pred2:
            images_status = {}
            for status, imgs in pred1.items():
                for img in imgs:
                    images_status.setdefault(img, {})["run1"] = status
            for status, imgs in pred2.items():
                for img in imgs:
                    images_status.setdefault(img, {})["run2"] = status

            changed = [
                img
                for img, d in images_status.items()
                if d.get("run1") != d.get("run2")
            ]
            diff_table = [
                {
                    "image": img,
                    runs[0]: images_status[img].get("run1", "-"),
                    runs[1]: images_status[img].get("run2", "-"),
                }
                for img in changed
            ]
            st.dataframe(pd.DataFrame(diff_table), use_container_width=True)
        else:
            st.warning("Missing prediction data for one or both runs.")

    def comparison_table(_self, runs: List[str], subset_key):
        """
        Compare a set of metrics for two runs
        Args
            runs : list of two run_ids to compare
            subset : key of df_split
        """

        df = _self.split_df[subset_key]
        # remove config data from the comparison
        metrics = sorted(
            set(df.columns) - {"model_path", "confidence", "max_bbox_size", "iou"}
        )
        data = []
        for metric in metrics:
            val1 = df.loc[df["run_id"] == runs[0], metric]
            val2 = df.loc[df["run_id"] == runs[1], metric]

            # TODO : deal differently for metrics for which lower is better (FP, FN)
            row = {
                "Metric": metric,
                runs[0]: f"**{val1}**" if val1 > val2 else val1,
                runs[1]: f"**{val2}**" if val2 > val1 else val2,
            }
            data.append(row)
        st.subheader(subset_key)
        st.markdown(
            "<style>table td {text-align: center;}</style>", unsafe_allow_html=True
        )
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    def score_graph(_self, metric):
        # Graph
        st.subheader("📈 Sequence F1-score per run")
        plt.figure(figsize=(10, 5))
        sns.barplot(data=_self.filtered_df, x="run_id", y=metric)
        plt.xticks(rotation=45)
        plt.ylabel("F1-score (sequence)")
        plt.xlabel("Run ID")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    def run(_self):
        st.title("Metrics")
        # Filters
        st.header("Filters")
        selected_runs = st.multiselect(
            "Run IDs", df["run_id"].unique(), default=df["run_id"].unique()
        )
        f1_min = st.slider("Filter by minimum F1", 0.0, 1.0, 0.0, 0.01)

        # Filter application
        filtered_df = df[df["run_id"].isin(selected_runs) & (df["seq_f1"] >= f1_min)]

        # Display Table
        st.subheader("Model Evaluation")
        st.dataframe(_self.df_model, use_container_width=True)

        st.subheader("Engine Evaluation")
        st.dataframe(_self.df_engine, use_container_width=True)

        # Graph
        _self.score_graph(metric="model_f1")

        # Compare 2 runs
        df = df.set_index("run_id")
        st.header("\ud83d\udd04 Compare Runs")

        compare_runs = st.checkbox("Enable run comparison")

        if compare_runs:
            run_choices = df.index.tolist()
            run1 = st.selectbox("Select first run", run_choices, key="run1")
            run2 = st.selectbox("Select second run", run_choices, key="run2")

            if run1 and run2 and run1 != run2:
                # Extract and compare metrics
                _self.comparison_table([run1, run2], "Model")
                _self.comparison_table([run1, run2], "Engine")

                # Compare predictions
                _self.compare_predictions([run1, run2])


if __name__ == "__main__":
    run_dirs = [
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1713-7016",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1748-7456",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1824-7189",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1859-2784",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1934-6024",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-2009-9189",
    ]

    app = App(run_dirs=run_dirs)
    app.run
