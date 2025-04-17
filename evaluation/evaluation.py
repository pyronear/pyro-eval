
from dataset import EvaluationDataset

class EvaluationPipeline:
    def __init__(self, dataset, parameters: dict = {}):
        self.dataset = dataset
        self.parameters = parameters # Engine config
        self.results = {}
        pass

    def model_evaluation(self):
        pass

    def engine_evaluation(self):
        pass

    def run(self):
        self.model_metrics = self.model_evaluation()
        self.engine_metrics = self.engine_evaluation()

if __name__ == "__main__":

    dataset_url = "https://huggingface.co/datasets/pyronear/pyro-sdis"
    dataset = EvaluationDataset(datapath=dataset_url)
    evaluation = EvaluationPipeline(dataset=dataset)