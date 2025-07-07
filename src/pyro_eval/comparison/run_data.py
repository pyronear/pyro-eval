import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        self.run_ids = [self.runs[0].run_id, self.runs[1].run_id]
        rows = []

        for image_id, models in status.items():
            status_A = models[self.run_ids[0]]
            status_B = models[self.run_ids[1]]
            row = {
                "Image Name": image_id,
                self.run_ids[0]: status_A,
                self.run_ids[1]: status_B,
                "Change Type" : self.get_change_type(status_A, status_B),
                'Transition': f"{status_A} → {status_B}" if status_A != status_B else status_A
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_change_type(self, status1: str, status2: str) -> str:
        """
        Determine the change between two statuses (improved or degraded)
        Semantic works only if we have a reference run
        """
        if status1 == status2:
            return 'unchanged'
        
        change = f"{status1}-to-{status2}"
        
        # Improvement
        if change in ['fn-to-tp', 'fp-to-tn']:
            return 'improved'
        
        # Degradations
        if change in ['tp-to-fn', 'tn-to-fp']:
            return 'degraded'
        
        return change

    def create_confusion_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Create an inter model confusion matrix 
        TODO: move out of the class
        """
        if df.empty:
            return go.Figure()
        
        # Create confusion matrix
        statuses = ['tp', 'fp', 'fn', 'tn']
        confusion_matrix = np.zeros((len(statuses), len(statuses)))
        
        for i, status1 in enumerate(statuses):
            for j, status2 in enumerate(statuses):
                count = len(df[(df[self.run_ids[0]] == status1) & (df[self.run_ids[1]] == status2)])
                confusion_matrix[i, j] = count
        
        # Créer la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=[s.upper() for s in statuses],
            y=[s.upper() for s in statuses],
            colorscale='Blues',
            text=confusion_matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion matrix",
            xaxis_title=self.run_ids[1],
            yaxis_title=self.run_ids[0],
            width=500,
            height=500
        )
        
        return fig

    def create_change_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates a graph of change distribution
        TODO: move out of the class
        """
        if df.empty:
            return go.Figure()
        
        change_counts = df['Change Type'].value_counts()
        
        # Colors for different types of changes
        colors = {
            'unchanged': '#95a5a6',
            'improved': '#27ae60',
            'degraded': '#e74c3c',
            'fp-to-tn': '#2ecc71',
            'fn-to-tp': '#2ecc71',
            'tp-to-fn': '#e67e22',
            'tn-to-fp': '#e67e22'
        }
        
        bar_colors = [colors.get(change, '#3498db') for change in change_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=change_counts.index,
                y=change_counts.values,
                marker_color=bar_colors,
                text=change_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Distribution of status change",
            xaxis_title="Change type",
            yaxis_title="Number of images",
            xaxis_tickangle=-45
        )
        
        return fig

    def display_status_badge(self, status: str) -> str:
        """Add color badge for status"""
        return f'<span class="status-badge status-{status}">{status.upper()}</span>'

    def __iter__(self):
        return iter(self.runs)
