import logging
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from pyroengine.engine import Engine

from dataset import EvaluationDataset
from data_structures import Sequence
from utils import make_dict_json_compatible, export_model, delete_tmp_model

class EngineEvaluator:
    # TODO : as EngineEvaluator and ModelEvaluator share some attributes and methods the should inherits from an EvaluatorClass
    # that manages instanciation, saving, run_id etc.
    def __init__(self,
                 dataset: EvaluationDataset,
                 config: dict = {},
                 save: bool = False,
                 run_id: str = None,
                 resume : bool = True
                 ):
        self.dataset = dataset
        self.config = config
        self.save = save # If save is True we regularly dump results and the config used
        self.run_id = run_id if len(run_id) > 0 else self.generate_run_id()
        self.resume = resume # If True, we look for partial results in results/<run_id>
        self.results_data = ["sequence_id", "image", "sequence_label", "image_label", "prediction", "conf", "timedelta"]
        self.predictions_csv = ""

    def run_engine_sequence(self, sequence:Sequence):
        """
        Instanciate an Engine and run predictions on a Sequence containing a list of images.
        Returns a dataframe containing image info and the confidence predicted
        """
        
        # Initialize a new Engine for each sequence
        # TODO : better handle default values
        
        model_path = self.config.get("model_path", None)
        needs_deletion  = False

        # We need to convert .pt local paths to .onnx as that's the only format supported by the engine
        if model_path.endswith(".pt"):
            model_path = export_model(model_path)
            needs_deletion = True

        try:
            pyroEngine = Engine(
                nb_consecutive_frames=self.config.get("nb_consecutive_frames", 4),
                conf_thresh=self.config.get("conf_thresh", 0.15),
                max_bbox_size=self.config.get("max_bbox_size", 0.4),
                model_path=self.config.get("model_path", None)
            )

            sequence_results = pd.DataFrame(columns=self.results_data)

            for image in sequence.images:
                pil_image = image.load()
                # Run prediction on a single image
                confidence = pyroEngine.predict(pil_image)
                sequence_results.loc[len(sequence_results)] = [
                    sequence.sequence_id,
                    image.image_path,
                    sequence.label,
                    image.label,
                    confidence > pyroEngine.conf_thresh,
                    confidence,
                    image.timedelta
                ]
        finally:
            if needs_deletion:
                delete_tmp_model(model_path)

        return sequence_results

    def run_engine_dataset(self):
        """
        Function that processes predictions through the Engine on sequences of images
        """

        if self.save:
            # Csv file where detailed predictions are dumped
            self.result_dir = os.path.join(os.path.dirname(__file__), "data/results", self.run_id)
            os.makedirs(self.result_dir, exist_ok=True)
            self.predictions_csv = os.path.join(self.result_dir, "results.csv")

        # Previous predictions are loaded if they exist and if resume is set to True
        # FIXME : this doesn't work predictions are re-run every time
        if os.path.isfile(self.predictions_csv) and self.resume:
            logging.info(f"Loading previous predictions in {self.predictions_csv}")
            self.predictions_df = pd.read_csv(self.predictions_csv)
        else:
            self.predictions_df = pd.DataFrame(columns=self.results_data)

        for sequence in self.dataset:
            if self.resume and sequence in set(self.predictions_df["sequence_id"].to_list()):
                logging.info(f"Results of {sequence} found in predictions csv, sequence skipped.")
                continue
            sequence_results = self.run_engine_sequence(sequence)
            
            # Add sequence results to result dataframe
            self.predictions_df = pd.concat([self.predictions_df, sequence_results])
            # Checkpoint to save predictions every 50 images
            if self.save and len(self.predictions_df) % 50 == 0:
                self.predictions_df.to_csv(self.predictions_csv, index=False)

        if self.save:
            logging.info(f"Saving predictions in {self.predictions_csv}")
            self.predictions_df.to_csv(self.predictions_csv, index=False)

    def compute_image_level_metrics(self):
        """
        Computes image-based metrics on the predicion dataframes.
        Those metrics do not take sequences into account.
        """
        y_true = self.predictions_df["image_label"].apply(lambda x: x != "")
        y_pred = self.predictions_df["prediction"]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        logging.info("Image-level metrics")
        logging.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        logging.info(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        return {
            "precision" : precision,
            "recall" : recall,
            "f1" : f1,
            "tn" : tn,
            "fp" : fp,
            "fn" : fn,
            "tp" : tp,
        }

    def compute_sequence_level_metrics(self):
        """
        Computes sequence-based metrics from the prediction dataframe
        """
        sequence_metrics = []

        for sequence_id, group in self.predictions_df.groupby("sequence_id"):
            sequence_label = group['sequence_label'].iloc[0]
            has_detection = group['prediction'].any()
            detection_timedeltas = group[group['prediction']]['timedelta']
            detection_timedeltas = pd.to_timedelta(detection_timedeltas, errors='coerce')
            detection_delay = detection_timedeltas.min() if not detection_timedeltas.empty else None

            if not detection_timedeltas.empty:
                detection_delay = np.min(detection_timedeltas)
            else:
                detection_delay = None

            sequence_metrics.append({
                'sequence_id': sequence_id,
                'label': sequence_label,
                'has_detection': has_detection,
                'detection_delay': detection_delay
            })

        sequence_df = pd.DataFrame(sequence_metrics)

        y_true_sequence = sequence_df['label']
        y_pred_sequence = sequence_df['has_detection']

        sequence_precision = precision_score(y_true_sequence, y_pred_sequence)
        sequence_recall = recall_score(y_true_sequence, y_pred_sequence)
        sequence_f1 = f1_score(y_true_sequence, y_pred_sequence)

        tp_sequences = sequence_df[(sequence_df['label'] == True) & (sequence_df['has_detection'] == True)]
        fn_sequences = sequence_df[(sequence_df['label'] == True) & (sequence_df['has_detection'] == False)]
        fp_sequences = sequence_df[(sequence_df['label'] == False) & (sequence_df['has_detection'] == True)]
        tn_sequences = sequence_df[(sequence_df['label'] == False) & (sequence_df['has_detection'] == False)]

        logging.info("Sequence-level metrics")
        logging.info(f"Precision: {sequence_precision:.3f}, Recall: {sequence_recall:.3f}, F1: {sequence_f1:.3f}")
        logging.info(f"TP: {len(tp_sequences)}, FP: {len(fp_sequences)}, FN: {len(fn_sequences)}, TN: {len(tn_sequences)}")

        if not tp_sequences['detection_delay'].isnull().all():
            avg_detection_delay = tp_sequences['detection_delay'].dropna().mean()
            logging.info(f"Avg. delay before detection (TP sequences): {avg_detection_delay}")
        else:
            logging.info("No detection delay info available for TP sequences.")
        
        return {
            "precision" : sequence_precision,
            "recall" : sequence_recall,
            "f1" : sequence_f1,
            "tp": len(tp_sequences),
            "fp": len(fp_sequences),
            "fn": len(fn_sequences),
            "tn": len(tn_sequences),
            "avg_detection_delay": avg_detection_delay if not tp_sequences['detection_delay'].isnull().all() else None
        }

    def evaluate(self):

        # Run Engine predictions on each sequence of the dataset
        self.run_engine_dataset()

        # Compute metrics from predictions
        self.metrics = {
            "run_id" : self.run_id,
            "image_metrics" : self.compute_image_level_metrics(),
            "sequence_metrics" : self.compute_sequence_level_metrics(),
        }

        # Save metrics in a json file
        if self.save:
            with open(os.path.join(self.result_dir, "engine_metrics.json"), 'w') as fip:
                json.dump(make_dict_json_compatible(self.metrics), fip)
    
    def generate_run_id(self):
        """
        Generates a unique run_id to store results
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rand_suffix = random.randint(1000, 9999)
        return f"run-{timestamp}-{rand_suffix}"