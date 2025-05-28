import json
import logging
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

credentials = (
    "/Users/theocayla/Documents/Dev/GoogleAPI/pyro-metrics-459816-9ef5ab17ce1c.json"
)


def build_dataframe(run_dirs, csv_path=None):
    rows = []
    for run_dir in run_dirs:
        json_file = Path(run_dir) / "metrics.json"
        with open(json_file, "r") as f:
            data = json.load(f)

        run_id = data.get("run_id")
        config = data.get("config", {})
        model_metrics = data.get("model_metrics", {})
        seq_metrics = data.get("engine_metrics", {}).get("sequence_metrics", {})
        img_metrics = data.get("engine_metrics", {}).get("image_metrics", {})
        dataset_info = data.get("dataset")

        rows.append(
            {
                "run_id": run_id,
                "model_path": config.get("model_path"),
                "conf_thresh": config.get("conf_thresh"),
                "nb_consecutive_frames": config.get("nb_consecutive_frames"),
                "iou": config.get("iou"),
                "max_bbox_size": config.get("max_bbox_size"),
                "model_precision": model_metrics.get("precision"),
                "model_recall": model_metrics.get("recall"),
                "model_f1": model_metrics.get("f1"),
                "model_fp": model_metrics.get("fp"),
                "model_tp": model_metrics.get("tp"),
                "model_fn": model_metrics.get("fn"),
                "seq_precision": seq_metrics.get("precision"),
                "seq_recall": seq_metrics.get("recall"),
                "seq_f1": seq_metrics.get("f1"),
                "seq_fp": seq_metrics.get("fp"),
                "seq_tp": seq_metrics.get("tp"),
                "seq_fn": seq_metrics.get("fn"),
                "avg_detection_delay" : seq_metrics.get("avg_detection_delay"),
                "img_precision": img_metrics.get("precision"),
                "img_recall": img_metrics.get("recall"),
                "img_f1": img_metrics.get("f1"),
                "img_fp": img_metrics.get("fp"),
                "img_tp": img_metrics.get("tp"),
                "img_fn": img_metrics.get("fn"),
                "dataset_hash" : dataset_info.get("hash"),
                "dataset_ID" : dataset_info.get("ID"),
            }
        )

    # sort using f1
    rows = sorted(rows, key=lambda r: r["seq_f1"] or 0, reverse=True)

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path)
    return df


def vizualize(df):
    sns.barplot(data=df, x="run_id", y="f1")
    plt.title("Comparaison des F1-score par run")
    plt.ylabel("F1-score")
    plt.xlabel("Run ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def export_google_sheet(df, sheet_name, key_column="run_id"):
    """
    Dumps a csv uin a google sheet
    If the sheet already exists, update the data and upload
    Use key_column as an identifier to see what's changed
    """
    # Google API Atuthentification
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials, scope)
    client = gspread.authorize(creds)

    # Open google sheet
    try:
        spreadsheet = client.open(sheet_name)
    except Exception as e:
        logging.error(f"Unable to open Google sheet : {e}")

    df[key_column] = df[key_column].astype(str)

    config_cols = [
        "run_id",
        "model_path",
        "dataset_ID",
        "dataset_hash"
    ]

    model_cols = [
        "run_id",
        "model_path",
        "model_precision",
        "model_recall",
        "model_f1",
        "model_fp",
        "model_tp",
        "model_fn",
    ]

    engine_cols = [
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

    df_model = df[model_cols].copy()
    df_engine = df[engine_cols].copy()
    df_config = df[config_cols].copy()
    logging.info("Updating google sheet")

    def update_worksheet(sheet_name, new_df, key_column="run_id"):
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=200, cols=50)

        # Retrieve existing content
        try:
            existing_df = get_as_dataframe(worksheet).dropna(how="all")
            existing_df[key_column] = existing_df[key_column].astype(str)
        except Exception:
            existing_df = pd.DataFrame(columns=new_df.columns)

        new_df[key_column] = new_df[key_column].astype(str)

        # Filter old dataframe to remove runs present in the new dataframe
        filtered_existing = existing_df[
            ~existing_df[key_column].isin(new_df[key_column])
        ]

        # Concatenate new dataframe to the old one
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)

        # Write everything in the sheet
        worksheet.clear()
        set_with_dataframe(worksheet, final_df)

        nb_updated = len(existing_df) - len(filtered_existing)
        nb_added = len(final_df) - len(existing_df)
        logging.info(f"{sheet_name} sheet updated")
        return (nb_updated, nb_added)

    update_worksheet("Config", df_config)
    update_worksheet("Model", df_model)
    nb_updated, nb_added = update_worksheet("Engine", df_engine)
    logging.info(f"{nb_added} runs added.")
    logging.info(f"{nb_updated} runs updated.")


if __name__ == "__main__":
    run_dirs = [
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1713-7016",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1748-7456",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1824-7189",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1859-2784",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-1934-6024",
        "/Users/theocayla/Documents/Dev/Pyronear/vision-rd/evaluation/data/results/run-20250514-2009-9189",
    ]
    comparison = "/Users/theocayla/Documents/Dev/Pyronear/debug/pipeline_runs/comparison_20250514.csv"
    df = build_dataframe(run_dirs, csv_path=comparison)
    sheet_name = "Pyro Metrics"
    export_google_sheet(df, sheet_name, key_column="run_id")
