from dataset import EvaluationDataset
from data_structures import CustomImage

class ModelEvaluator:
    def __init__(self, dataset: EvaluationDataset, config={}, device=None):
        self.dataset = dataset
        self.config = config
        self.model_path = self.config.get("model_path", None)
        self.inference_params = self.config.get("inference_params", {})

    def inference(self, image: CustomImage):
        # predict
        pass

    def evaluate(self):

        pass