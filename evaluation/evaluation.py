from dataset import EvaluationDataset
from engine_evaluation import EngineEvaluator
from model_evaluation import ModelEvaluator

class EvaluationPipeline:
    def __init__(self, dataset, config: dict = {}):
        self.dataset = dataset
        self.config = config # Engine config
        self.model_evaluator = ModelEvaluator(dataset, self.config["model_config"])
        self.engine_evaluator = EngineEvaluator(dataset, self.config["engine_config"])

    def run(self):
        self.model_metrics = self.model_evaluation()
        self.engine_metrics = self.engine_evaluation()

if __name__ == "__main__":

    dataset_url = "https://huggingface.co/datasets/pyronear/pyro-sdis"
    dataset = EvaluationDataset(datapath=dataset_url)
    evaluation = EvaluationPipeline(dataset=dataset)