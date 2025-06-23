import os
import json
from pathlib import Path
from typing import List


class RunData:
    def __init__(self, run_id):
        self.run_id = run_id
        self.filepath = self.get_filepath()
        self.data = self.load_data()
        self.engine_metrics = self.data.get("engine_metrics", {})
        self.model_metrics = self.data.get("model_metrics", {})
        self.config = self.data.get("config", {})
        self.dataset = self.data.get("dataset", {})
    
    def get_filepath(self):
        filepath = Path(os.path.dirname(__file__)) / "src/pyro_eval/data/evaluation/{self.run_id}/metrics.json"
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Data file not found : {filepath}")
        else:
            return filepath

    def load_data(self):
        try:
            with open(self.filepath, 'r') as fp:
                data = json.load(fp)
            return data
        except:
            raise ValueError(f"Unable to load data from {self.filepath}")


class RunComparison:

    def __init__(self, runs: List[RunData]):
        self.runs = runs

    def compare_predictions(self):
        """
        Analyzes each run predictions and returns changes in image status
        """
        pass

    def __iter__(self):
        return iter(self.runs)
