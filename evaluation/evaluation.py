import json
import logging
import os
import random
from datetime import datetime
from typing import List

from dataset import EvaluationDataset
from engine_evaluation import EngineEvaluator
from model_evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EvaluationPipeline:
    def __init__(self,
                 dataset: EvaluationDataset,
                 config: dict = {},
                 save: bool = False,
                 run_id: str = "",
                 resume: bool = False,
                 device : str = None,
                 eval : List[str] = ["model", "engine"]):

        self.dataset = dataset
        self.config = config # Engine config
        self.save = save
        self.run_id = run_id or self.generate_run_id()
        self.resume = resume
        self.eval = eval # list of evaluation to perform, all by default
        self.metrics = {}

        # Evaluate the model performance on single images
        self.model_evaluator = ModelEvaluator(dataset, self.config, device)
        
        # Evaluate the engine performance on series of images
        self.engine_evaluator = EngineEvaluator(dataset,
                                                config=self.config,
                                                save=self.save,
                                                run_id=self.run_id,
                                                resume=self.resume)

    def run(self):
        if "model" in self.eval:
            logging.info("Compute model metrics")
            self.metrics["model_metrics"] = self.model_evaluator.evaluate()
            self.display_metrics(subset=["model"])
        if "engine" in self.eval:
            logging.info("Compute engine metrics")
            self.metrics["engine_metrics"] = self.engine_evaluator.evaluate()

        self.display_metrics()

        if self.save:
            self.save_metrics()

    def save_metrics(self):
        """
        SAve results in a json file
        """
        result_file = f"data/results/{self.run_id}/metrics.json"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        logging.info(f"Saving metrics in {result_file}")

        self.metrics.update({
            "config" : self.config
        })

        with open(result_file, 'w') as fp:
            json.dump(self.metrics, fp)

    def display_metrics(self, subset = ["model", "engine"]):
        def format_metric(value):
            return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"

        model_metrics = self.metrics.get("model_metrics", {})
        logging.info(f"Run ID: {self.run_id}")
        if "model" in subset:
            logging.info("Model Metrics:")
            logging.info(f"  Precision: {format_metric(model_metrics.get('precision', 'N/A'))}")
            logging.info(f"  Recall:    {format_metric(model_metrics.get('recall', 'N/A'))}")
            logging.info(f"  F1 Score:  {format_metric(model_metrics.get('f1_score', 'N/A'))}")
            logging.info(f"  False positives:  {format_metric(model_metrics.get('fp', 'N/A'))}")
            logging.info(f"  True positives:  {format_metric(model_metrics.get('tp', 'N/A'))}")
            logging.info(f"  False negatives:  {format_metric(model_metrics.get('fn', 'N/A'))}")

        engine_image_metrics = self.metrics.get("engine_metrics", {}).get("image_metrics", {})
        engine_sequence_metrics = self.metrics.get("engine_metrics", {}).get("sequence_metrics", {})
        if "engine" in subset:
            logging.info("Engine Metrics:")
            logging.info("    Image Metrics:")
            logging.info(f"     Precision: {format_metric(engine_image_metrics.get('precision', 'N/A'))}")
            logging.info(f"     Recall:    {format_metric(engine_image_metrics.get('recall', 'N/A'))}")
            logging.info(f"     F1 Score:  {format_metric(engine_image_metrics.get('f1_score', 'N/A'))}")
            logging.info("    Sequence Metrics:")
            logging.info(f"     Precision: {format_metric(engine_sequence_metrics.get('precision', 'N/A'))}")
            logging.info(f"     Recall:    {format_metric(engine_sequence_metrics.get('recall', 'N/A'))}")
            logging.info(f"     F1 Score:  {format_metric(engine_sequence_metrics.get('f1_score', 'N/A'))}")
            logging.info(f"     Averagde Detecion Delay:  {format_metric(engine_sequence_metrics.get('avg_detection_delay', 'N/A'))}")

    def generate_run_id(self):
        """
        Generates a unique run_id to store results
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rand_suffix = random.randint(1000, 9999)
        return f"run-{timestamp}-{rand_suffix}"

if __name__ == "__main__":

    # Usage example
    
    # Instanciate Dataset
    dataset_path = "" # Folders with two sub-folders : images and labels
    dataset = EvaluationDataset(datapath=dataset_path)

    # Launch Evaluation
    config = {
        "model_path" : "data/models/250318_yolo11/yolov11s.pt"
    }
    evaluation = EvaluationPipeline(dataset=dataset, config=config, save=True, device="mps")
    evaluation.run()