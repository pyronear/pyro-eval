import logging

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
                 resume: bool = False):

        self.dataset = dataset
        self.config = config # Engine config
        self.save = save
        self.run_id = run_id
        self.resume = resume
        
        # Evaluate the model performance on single images
        self.model_evaluator = ModelEvaluator(dataset, self.config.get("model_config", {}))
        
        # Evaluate the engine performance on series of images
        self.engine_evaluator = EngineEvaluator(dataset,
                                                config=self.config.get("engine_config", {}),
                                                save=self.save,
                                                run_id=self.run_id,
                                                resume=self.resume)

    def run(self):
        self.model_metrics = self.model_evaluator.evaluate()
        self.engine_metrics = self.engine_evaluator.evaluate()

    def display_metrics(self):
        pass

if __name__ == "__main__":

    # Usage example
    
    # Instanciate Dataset
    dataset_path = "" # Folders with two sub-folders : images and labels
    dataset = EvaluationDataset(datapath=dataset_path)

    # Launch Evaluation
    evaluation = EvaluationPipeline(dataset=dataset, save=True)
    evaluation.run()