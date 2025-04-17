from dataset import EvaluationDataset

class ModelEvaluator:
    def __init__(self, dataset: EvaluationDataset, config={}):
        self.dataset = dataset
        self.config = config

    def evaluate(self):
        pass