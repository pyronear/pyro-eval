from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

from pyro_eval.dataset import EvaluationDataset
from pyro_eval.evaluation import EvaluationPipeline


def run_grid_search(out_file, loaded_df=None):
    # Usage example

    # Instanciate Dataset
    model_datapath = "data/datasets/1.3.5/wildfire_test_1.3.5"  # Folders with two sub-folders : images and labels
    engine_datapath = "data/datasets/1.3.5/wildfire_temporal_test_1.3.5"  # Folders with two sub-folders : images and labels

    datasets = {
        "model": EvaluationDataset(
            datapath=model_datapath, dataset_ID="wildfire_test_v1.3.5"
        ),
        "engine": EvaluationDataset(
            datapath=engine_datapath, dataset_ID="temporal_dataset_v1.3.5"
        ),
    }

    configs = [
        {
            "model_path": model_path,
            "engine": {
                "nb_consecutive_frames": nb_consecutive_frames,
                "conf_thresh": conf_thresh,
                "max_bbox_size": max_bbox_size,
            },
            "model": {
                "conf": conf,
                # "iou" : iou,
            },
        }
        for model_path in ["data/models/yolo11s_colorful-chameleon_v3.0.0_7bd9f32.pt"]
        for nb_consecutive_frames in [2, 3, 4, 5, 6, 7, 8]
        for conf_thresh in [0.05, 0.09, 0.13, 0.15, 0.19]
        for conf in [0.05]
        for max_bbox_size in [0.2, 0.3, 0.4, 0.5]
    ]

    columns = [
        "model_path",
        "nb_consecutive_frames",
        "conf_thresh",
        "max_bbox_size",
        "conf",
        "iou",
        "engine_f1",
        "engine_prec",
        "engine_rec",
        "model_f1",
        "model_prec",
        "model_rec",
        "config_str",
    ]

    df = pd.DataFrame(columns=columns)

    for i, config in enumerate(configs):
        # Identify each run by a string describing the config
        config_str = f"{config['engine']['nb_consecutive_frames']}_{config['engine']['conf_thresh']}_{config['engine']['max_bbox_size']}"
        if loaded_df is not None and config_str in loaded_df["config_str"]:
            # Load previous metrics in the dataframe provided
            seq_f1 = loaded_df.loc[loaded_df["config_str"] == config_str, "engine_f1"]
            seq_prec = loaded_df.loc[
                loaded_df["config_str"] == config_str, "engine_prec"
            ]
            seq_rec = loaded_df.loc[loaded_df["config_str"] == config_str, "engine_rec"]
            model_f1 = loaded_df.loc[loaded_df["config_str"] == config_str, "model_f1"]
            model_prec = loaded_df.loc[
                loaded_df["config_str"] == config_str, "model_prec"
            ]
            model_rec = loaded_df.loc[
                loaded_df["config_str"] == config_str, "model_rec"
            ]
        else:
            evaluation = EvaluationPipeline(
                dataset=datasets,
                config=config,
                device="mps",
                use_existing_predictions=True,
            )
            metrics = evaluation.run()
            evaluation.save_metrics(Path("data/evaluation"))
            seq_f1 = metrics["engine_metrics"]["sequence_metrics"]["f1"]
            seq_prec = metrics["engine_metrics"]["sequence_metrics"]["precision"]
            seq_rec = metrics["engine_metrics"]["sequence_metrics"]["recall"]
            model_f1 = metrics["model_metrics"]["f1"]
            model_prec = metrics["model_metrics"]["precision"]
            model_rec = metrics["model_metrics"]["recall"]

        df.loc[len(df)] = [
            config["model_path"],
            config["engine"]["nb_consecutive_frames"],
            config["engine"]["conf_thresh"],
            config["engine"]["max_bbox_size"],
            config["model"]["conf"],
            config["model"]["iou"],
            seq_f1,
            seq_prec,
            seq_rec,
            model_f1,
            model_prec,
            model_rec,
            config_str,
        ]
        if i % 10 == 0:
            df.to_csv(out_file)

    df.to_csv(out_file)


def analyze_results(df):
    param_columns = [
        "nb_consecutive_frames",
        "conf_thresh",
        "conf",
        "max_bbox_size",
    ]

    # Style seaborn
    sns.set_theme(style="whitegrid")

    # Parameter plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, param in enumerate(param_columns):
        ax = axes[i // 3][i % 3]
        sns.lineplot(data=df.sort_values(param), x=param, y="engine_f1", ax=ax)
        ax.set_title(f"F1 score vs {param}")
    plt.tight_layout()
    plt.show()

    # F1 histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df["engine_f1"], bins=20, kde=True, color="skyblue")
    plt.title("F1 Distribution")
    plt.xlabel("F1 score")
    plt.ylabel("Number of runs")
    plt.grid(True)
    plt.show()

    # Parallel coordinates
    df_vis = df.copy()
    df_vis["f1_group"] = pd.qcut(df["engine_f1"], q=5, labels=False)
    plt.figure(figsize=(12, 6))
    parallel_coordinates(
        df_vis[param_columns + ["f1_group"]],
        class_column="f1_group",
        colormap=plt.cm.viridis,
    )
    plt.title("Parallel coordinates")
    plt.show()

    # Check this representation : https://github.com/scikit-learn/scikit-learn/issues/24281


if __name__ == "__main__":
    file = ""
    loaded_df = pd.read_csv(file)

    out_file = ""
    run_grid_search(out_file=out_file, loaded_df=loaded_df)

    df = pd.read_csv(out_file)
    top_scores = df.nlargest(10, "engine_f1")
    print(top_scores)
    analyze_results(df)
