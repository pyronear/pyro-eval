import json
import logging
import os
from glob import glob
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

current_dir = os.path.dirname(__file__)
credentials_path = os.path.abspath(
    os.path.join(current_dir, "..", "credentials", "pyro-metrics-creds.json")
)

sheets_config = {
    "Config": [
        "Run ID",
        "Model Path",
        "Model Dataset ID",
        "Model Dataset Hash",
        "Model Hash",
        "Number of images (Model dataset)",
        "Engine Dataset ID",
        "Engine Dataset Hash",
        "Number of images (Engine dataset)",
        "Number of sequences (Engine dataset)",
    ],
    "Model": [
        "Run ID",
        "Model Precision",
        "Model Recall",
        "Model F1",
        "Model FP",
        "Model TP",
        "Model FN",
        "Model TN",
        "Model Path",
        "Model Conf",
        "Model IoU",
        "Model Imgsz",
        "Model Dataset Hash",
        "Model Dataset ID",
    ],
    "Engine": [
        "Run ID",
        "Sequence Precision",
        "Sequence Recall",
        "Sequence F1",
        "Sequence FP",
        "Sequence TP",
        "Sequence FN",
        "Sequence TN",
        "Avg Detection Delay",
        "Image Precision",
        "Image Recall",
        "Image F1",
        "Image FP",
        "Image TP",
        "Image FN",
        "Model Path",
        "Engine Conf thresh",
        "Engine Nb consecutive frames",
        "Engine Max Bbox Size",
        "Engine Dataset Hash",
        "Engine Dataset ID",
    ],
}


def build_dataframe(run_dirs, csv_path=None):
    rows = []

    for run_dir in run_dirs:
        json_file = Path(run_dir) / "metrics.json"
        with open(json_file, "r") as f:
            data = json.load(f)
        config = data.get("config", {})
        model_config = config.get("model")
        engine_config = config.get("engine")
        model_metrics = data.get("model_metrics", {})
        seq_metrics = data.get("engine_metrics", {}).get("sequence_metrics", {})
        img_metrics = data.get("engine_metrics", {}).get("image_metrics", {})
        dataset_info = data.get("dataset")

        rows.append(
            {
                # Config
                "Run ID": data.get("run_id"),
                "Model Path": config.get("model_path").split("pyro-eval")[-1],
                "Engine Conf thresh": engine_config.get("conf_thresh"),
                "Engine Nb consecutive frames": engine_config.get(
                    "nb_consecutive_frames"
                ),
                "Model IoU": model_config.get("iou"),
                "Model Imgsz": model_config.get("imgsz"),
                "Model Conf": model_config.get("conf"),
                "Model Hash": model_config.get("hash"),
                "Engine Max Bbox Size": engine_config.get("max_bbox_size"),
                "Engine Dataset Hash": dataset_info.get("engine", {}).get("hash"),
                "Engine Dataset ID": dataset_info.get("engine", {}).get("ID"),
                "Number of images (Engine dataset)": dataset_info.get("engine", {}).get(
                    "Number of images"
                ),
                "Number of sequences (Engine dataset)": dataset_info.get(
                    "engine", {}
                ).get("Number of sequences"),
                "Model Dataset Hash": dataset_info.get("model", {}).get("hash"),
                "Model Dataset ID": dataset_info.get("model", {}).get("ID"),
                "Number of images (Model dataset)": dataset_info.get("model", {}).get(
                    "Number of images"
                ),
                # Model metrics
                "Model Precision": model_metrics.get("precision"),
                "Model Recall": model_metrics.get("recall"),
                "Model F1": model_metrics.get("f1"),
                "Model FP": model_metrics.get("fp"),
                "Model TP": model_metrics.get("tp"),
                "Model FN": model_metrics.get("fn"),
                "Model TN": model_metrics.get("tn"),
                # Engine metrics
                "Sequence Precision": seq_metrics.get("precision"),
                "Sequence Recall": seq_metrics.get("recall"),
                "Sequence F1": seq_metrics.get("f1"),
                "Sequence FP": seq_metrics.get("fp"),
                "Sequence TP": seq_metrics.get("tp"),
                "Sequence FN": seq_metrics.get("fn"),
                "Sequence TN": seq_metrics.get("tn"),
                "Avg Detection Delay": seq_metrics.get("avg_detection_delay"),
                "Image Precision": img_metrics.get("precision"),
                "Image Recall": img_metrics.get("recall"),
                "Image F1": img_metrics.get("f1"),
                "Image FP": img_metrics.get("fp"),
                "Image TP": img_metrics.get("tp"),
                "Image FN": img_metrics.get("fn"),
            }
        )

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path)
    return df


class GoogleSheetExporter:
    def __init__(self, spreadsheet_name):
        self.spreadsheet_name = spreadsheet_name
        self.client = self._authenticate()
        self.spreadsheet = self._open_spreadsheet()

    def _authenticate(self):
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_path, scope
        )
        return gspread.authorize(creds)

    def _open_spreadsheet(self):
        try:
            return self.client.open(self.spreadsheet_name)
        except Exception as e:
            logging.error(f"Unable to open Google Sheet: {e}")
            raise

    def clear_sheets(self, columns):
        for worksheet_name, columns in sheets_config.items():
            try:
                worksheet = self.spreadsheet.worksheet(worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                worksheet = self.spreadsheet.add_worksheet(
                    title=worksheet_name, rows=200, cols=50
                )
            worksheet.clear()
            set_with_dataframe(worksheet, pd.DataFrame(columns=columns))
            logging.info(f"{worksheet_name} cleared.")

    def update_sheet(self, worksheet_name, new_df, key_column="Run ID"):
        try:
            worksheet = self.spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = self.spreadsheet.add_worksheet(
                title=worksheet_name, rows=200, cols=50
            )

        try:
            existing_df = get_as_dataframe(worksheet).dropna(how="all")
            existing_df[key_column] = existing_df[key_column].astype(str)
        except Exception:
            existing_df = pd.DataFrame(columns=new_df.columns)

        new_df.loc[:, key_column] = new_df[key_column].astype(str)
        filtered_existing = existing_df[
            ~existing_df[key_column].isin(new_df[key_column])
        ]
        final_df = pd.concat([filtered_existing, new_df], ignore_index=True)

        worksheet.clear()
        set_with_dataframe(worksheet, final_df)

        nb_updated = len(existing_df) - len(filtered_existing)
        nb_added = len(final_df) - len(existing_df)
        logging.info(
            f"{worksheet_name} updated: {nb_added} added, {nb_updated} updated"
        )
        return nb_added, nb_updated

    def export_dataframe(self, df):
        df["Run ID"] = df["Run ID"].astype(str)
        for sheet_name, columns in sheets_config.items():
            self.update_sheet(sheet_name, df[columns])


def create_spreadsheet(spreadsheet_name, email_adress=None):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # Authentification
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    gc = gspread.authorize(creds)

    spreadsheet = gc.create(spreadsheet_name)

    if email_adress is not None:
        spreadsheet.share(email_address=email_adress, perm_type="user", role="writer")


if __name__ == "__main__":
    eval_dir = os.path.abspath(os.path.join(current_dir, "..", "data/evaluation"))

    run_dirs = [
        run
        for run in glob(f"{eval_dir}/*")
        if "run-20250626-13" in run or "run-20250626-14" in run
    ]
    # run_dirs += [run for run in glob(f"{eval_dir}/*") if "run-20250624-13" in run]

    df = build_dataframe(run_dirs, csv_path=None)
    exporter = GoogleSheetExporter("Pyro Metrics")

    # Remove old data (optionnal)
    exporter.clear_sheets(df.columns)

    # Upload new data
    exporter.export_dataframe(df)

    # Update archive
    archive_exporter = GoogleSheetExporter("Pyro Metrics Archive")
    archive_exporter.export_dataframe(df)
