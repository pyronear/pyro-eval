import logging
import json
import os

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from pyroengine.engine import Engine

from utils import is_image, replace_bool_values, compute_f1_score, generate_run_id

from dataset import EvaluationDataset
from data_structures import CustomImage, Session

class EngineEvaluator:
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
        self.run_id = run_id or generate_run_id()
        self.resume = resume # If True, we look for partial results in results/<run_id>
        self.results_data = ["session", "image", "session_label", "image_label", "prediction", "conf"]

    def run_engine_session(self, session:Session):
        """
        Instanciate an Engine and run predictions on a Session containing a list of images.
        Returns a dataframe containing image info and the confidence predicted
        """
        
        # Initialize a new Engine for each session
        # TODO : better handle default values
        pyroEngine = Engine(
            nb_consecutive_frames=self.config.get("nb_consecutive_frames", 4),
            conf_thresh=self.config.get("conf_thresh", 0.15),
            max_bbox_size=self.config.get("max_bbox_size", 0.4),
            model_path=self.config.get("model_path", None)
        )

        session_results = pd.DataFrame(columns=self.results_data)

        for image in session.images:
            pil_image = image.load()
            confidence = pyroEngine.predict(pil_image)
            session_results.loc[len(session_results)] = [
                session.session_id,
                image.image_path,
                session.label,
                image.label,
                confidence > pyroEngine.conf_thresh,
                confidence
            ]

        return session_results


    def run_engine_dataset(self):
        """
        Function that processes predictions through the Engine on sessions of images
        """

        if self.save:
            # Csv file where detailed predictions are dumped
            self.result_dir = os.path.join("data/results", self.run_id)
            os.makedirs(self.result_dir, exist_ok=True)
            self.predictions_csv = os.path.join(self.result_dir, "results.csv")

        # Previous predictions are loaded if they exist and if resume is set to True
        if os.path.isfile(self.predictions_csv) and self.resume:
            self.predictions_df = pd.read_csv(self.predictions_csv)
        else:
            self.predictions_df = pd.DataFrame(columns=self.results_data)

        for session in self.dataset:
            if self.resume and session in set(self.predictions_df["session"].to_list()):
                logging.info(f"Results of {session} found in predictions csv, session skipped.")
                continue
            session_results = self.run_engine_session(session)
            
            # Add session results to result dataframe
            self.predictions_df = pd.concat(self.predictions_df, session_results)
            # Checkpoint to save predictions every 50 images
            if self.save and len(self.predictions_df) % 50 == 0:
                self.predictions_df.to_csv(self.predictions_csv, index=False)

        if self.save:
            self.predictions_df.to_csv(self.predictions_csv, index=False)

    def compute_image_level_metrics(self):
        """
        Computes image-based metrics on the predicion dataframes.
        Those metrics do not take sessions into account.
        """
        y_true = self.predictions_df["image_label"].apply(lambda x: x != "")
        y_pred = self.predictions_df["prediction"]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        logging.info("Image-level metrics")
        logging.info(f"Precision : {precision:.3f}")
        logging.info(f"Recall    : {recall:.3f}")
        logging.info(f"F1-score  : {f1:.3f}")
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

    def compute_session_level_metrics(self):
        """
        Computes session-based metrics from the prediction dataframe
        """
        session_metrics = []

        for session_id, group in self.predictions_df.groupby("session_id"):
            session_label = group['session_label'].iloc[0]
            has_detection = group['prediction'].any()
            detection_timedeltas = group[group['prediction']]['timedelta']
            detection_delay = detection_timedeltas.min() if not detection_timedeltas.empty else None

            session_metrics.append({
                'session_id': session_id,
                'label': session_label,
                'has_detection': has_detection,
                'detection_delay': detection_delay
            })

        session_df = pd.DataFrame(session_metrics)

        y_true_session = session_df['label']
        y_pred_session = session_df['has_detection']

        session_precision = precision_score(y_true_session, y_pred_session)
        session_recall = recall_score(y_true_session, y_pred_session)
        session_f1 = f1_score(y_true_session, y_pred_session)

        tp_sessions = session_df[(session_df['label'] == True) & (session_df['has_detection'] == True)]
        fn_sessions = session_df[(session_df['label'] == True) & (session_df['has_detection'] == False)]
        fp_sessions = session_df[(session_df['label'] == False) & (session_df['has_detection'] == True)]
        tn_sessions = session_df[(session_df['label'] == False) & (session_df['has_detection'] == False)]

        logging.info("Session-level metrics")
        logging.info(f"TP: {len(tp_sessions)}")
        logging.info(f"FP: {len(fp_sessions)}")
        logging.info(f"FN: {len(fn_sessions)}")
        logging.info(f"TN: {len(tn_sessions)}")
        logging.info(f"Precision: {session_precision:.3f}, Recall: {session_recall:.3f}, F1: {session_f1:.3f}")

        if not tp_sessions['detection_delay'].isnull().all():
            avg_detection_delay = tp_sessions['detection_delay'].dropna().mean()
            logging.info(f"Avg. delay before detection (TP sessions): {avg_detection_delay}")
        else:
            logging.info("No detection delay info available for TP sessions.")
        
        return {
            "precision" : session_precision,
            "recall" : session_recall,
            "f1" : session_f1,
            "tp": len(tp_sessions),
            "fp": len(fp_sessions),
            "fn": len(fn_sessions),
            "tn": len(tn_sessions),
            "avg_detection_delay": avg_detection_delay if not tp_sessions['detection_delay'].isnull().all() else None
        }

    def evaluate(self):
        self.metrics = {
            "image_metrics" : self.compute_image_level_metrics(),
            "session_metrics" : self.compute_session_level_metrics(),
        }
        
        if self.save:
            with open(os.path.join(self.result_dir, "metrics.json"), 'w') as fip:
                json.dump(replace_bool_values(self.metrics), fip)