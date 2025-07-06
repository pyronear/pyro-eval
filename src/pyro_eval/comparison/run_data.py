import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/pyro_eval/comparison/run_data.py → ../.. = pyro_eval

class RunData:
    def __init__(self, run_id):
        self.run_id = run_id
        self.filepath = self.get_filepath()
        self.data = self.load_data()
        self.engine_metrics = self.data.get("engine_metrics", {})
        self.model_metrics = self.data.get("model_metrics", {})
        self.config = self.data.get("config", {})
        self.dataset = self.data.get("dataset", {})
        self.engine_datapath = self.dataset.get("engine", {}).get("datapath")
        self.model_datapath = self.dataset.get("model", {}).get("datapath")

    def get_filepath(self) -> str:
        filepath = PROJECT_ROOT / f"data/evaluation/{self.run_id}/metrics.json"
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Data file not found : {filepath}")
        else:
            return filepath

    def load_data(self) -> Dict:
        try:
            with open(self.filepath, 'r') as fp:
                data = json.load(fp)
            return data
        except:
            raise ValueError(f"Unable to load data from {self.filepath}")

    def get_status_by_image(self) -> Dict[str, str]:
        """
        Reverse key value from the predictions status stored in metrics.json:
        {"fp" : [img_list], "fn" : [img_list], "tp" : [img_list], "tn" : [img_list]}
        -> {"img_i" : "fp", "img_j" : "tp", ...}
        """
        self.model_predictions = self.engine_metrics.get("predictions", {})
        status_by_image = {}
        for status, images in self.model_predictions.items():
            for img in images:
                status_by_image[img] = status
        return status_by_image

    def get_status_by_sequence(self) -> Dict[str, str]:
        """
        Reverse key value from the predictions status stored in metrics.json:
        {"fp" : [img_list], "fn" : [img_list], "tp" : [img_list], "tn" : [img_list]}
        -> {"img_i" : "fp", "img_j" : "tp", ...}
        """
        self.sequence_predictions = self.engine_metrics.get("sequence_metrics", {}).get("predictions", {})
        status_by_sequence = {}
        for status, sequences in self.sequence_predictions.items():
            for seq in sequences:
                status_by_sequence[seq] = status
        return status_by_sequence

class RunComparison:

    def __init__(self, runs: List[RunData]):
        self.runs = runs

    def compare_predictions(self, source: str) -> Dict[str, Dict[str, str]]:
        """
        Returns a dict:
        {
            'image1.jpg': {'model_A': 'fp', 'model_B': 'tp'},
            ...
        }
        """
        comparison = defaultdict(dict)
        for run in self.runs:
            if source == "model":
                run_status = run.get_status_by_image()
            elif source == "sequence":
                run_status = run.get_status_by_sequence()
            for img, status in run_status.items():
                comparison[img][run.run_id] = status
        return dict(comparison)

    def get_changed_status(self, source: str) -> Dict[str, Dict[str, str]]:
        """
        Filters comparison to keep only images with differing statuses between models
        """
        full_comparison = self.compare_predictions(source)
        changed = {img: statuses for img, statuses in full_comparison.items()
                   if len(set(statuses.values())) > 1}
        return changed

    def get_status_dataframe(self, status: Dict) -> pd.DataFrame:

        run_ids = [self.runs[0].run_id, self.runs[1].run_id]
        rows = []

        for image_id, models in status.items():
            row = {
                "image_name": image_id,
                run_ids[0]: models[run_ids[0]],
                run_ids[1]: models[run_ids[1]]
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def __iter__(self):
        return iter(self.runs)
