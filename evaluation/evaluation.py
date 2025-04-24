import json
import logging
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
        self.model_evaluator = ModelEvaluator(dataset, self.config.get("model_config", {}), device)
        
        # Evaluate the engine performance on series of images
        self.engine_evaluator = EngineEvaluator(dataset,
                                                config=self.config.get("engine_config", {}),
                                                save=self.save,
                                                run_id=self.run_id,
                                                resume=self.resume)

    def run(self):
        if "model" in self.eval:
            self.metrics["model_metrics"] = self.model_evaluator.evaluate()
            logging.info
        if "engine" in self.eval:
            self.metrics["engine_metrics"] = self.engine_evaluator.evaluate()

    def save_metrics(self):
        """
        SAve results in a json file
        """
        result_file = f"data/results/{self.run_id}/metrics.json"
        logging.info(f"Saving metrics in {result_file}")

        with open(self.result_file, 'w') as fp:
            json.dump(self.metrics, fp)

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
    evaluation = EvaluationPipeline(dataset=dataset, save=True, device="mps")
    evaluation.run()