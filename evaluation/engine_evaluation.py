import logging
import json
import os
import uuid
from datetime import datetime

import pandas as pd
from tqdm import tqdm 

from pyroengine.engine import Engine

from utils import is_image, replace_bool_values, compute_f1_score, generate_run_id

from dataset import EvaluationDataset
from data_structures import CustomImage, Session

class EngineEvaluator:
    def __init__(self,
                 dataset: EvaluationDataset,
                 config={},
                 save=False,
                 run_id = None,
                 resume = True
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
            data = pd.read_csv(self.predictions_csv)
        else:
            data = pd.DataFrame(columns=self.results_data)

        for session in self.dataset:
            if self.resume and session in set(data["session"].to_list()):
                logging.info(f"Results of {session} found in predictions csv, session skipped.")
                continue
            session_results = self.run_engine_session(session)
            
            # Add session results to result dataframe
            data = pd.concat(data, session_results)
            # Checkpoint to save predictions every 50 images
            if self.save and len(data) % 50 == 0:
                data.to_csv(self.predictions_csv, index=False)

        if self.save:
            data.to_csv(self.predictions_csv, index=False)

    def compute_metrics(self):
        # average detection time
        # TP, FP, TN, FN
        # Precision, Recall
        # F1
        '''
        Computes metrics on the predictions run, from results stored in the prediction csv.
        '''
        results = {}
        df = pd.read_csv(os.path.join(datapath, "merged_preds.csv")) 
        predictions = [column for column in df.columns if "prediction_" in column]
        nb_fire = 0
        nb_non_fire = 0
        for session in set(df["session"].to_list()):
            if df.loc[df["session"] == session, "session_label"].unique()[0] == True:
                nb_fire += 1
            else:
                nb_non_fire += 1
        with open(os.path.join(datapath, "config.json"), 'r') as fp:
            dConfigs = json.load(fp)

        for predictionId in predictions:
            results[predictionId] = {
                "details" : {},
                "missed_detections" : 0,
                "false_positives" : 0,
                "config" : dConfigs.get(predictionId.split("_")[-1], {}),
            }

            for session in set(df["session"].to_list()):
                session_data = df[df["session"] == session]
                label_value = session_data["session_label"].iloc[0]
                prediction_value = session_data[predictionId].any()

                if prediction_value != label_value:
                    if label_value:
                        results[predictionId]["missed_detections"] += 1
                        results[predictionId]["details"].update({session : "Missed detection"})
                    else:
                        results[predictionId]["false_positives"] += 1
                        results[predictionId]["details"].update({session : "False positive"})
                else:
                    if label_value:
                        results[predictionId]["details"].update({session : "Correct : Fire detectedL."})
                    else:
                        results[predictionId]["details"].update({session : "Correct : No fire detected."})

            results[predictionId]["f1"] = compute_f1_score(
                tp=nb_fire - results[predictionId]["missed_detections"],
                fp=results[predictionId]["false_positives"],
                fn=results[predictionId]["missed_detections"]
            )

            results[predictionId]["missed_detections"] = round(results[predictionId]["missed_detections"]/nb_fire, 2)
            results[predictionId]["false_positives"] = round(results[predictionId]["false_positives"]/nb_non_fire, 2)

        with open(os.path.join(datapath, "results_merged.json"), 'w') as fp:
            json.dump(replace_bool_values(results), fp)


    def evaluate(self):
        # TODO : add metric detection time (time spent between first image and first detection in a session)
        pass