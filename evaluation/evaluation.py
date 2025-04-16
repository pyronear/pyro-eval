
class EvaluationPipeline:
    def __init__(self, parameters: dict = {}):
        self.parameters = parameters # Engine config
        pass

    def model_evaluation(self):
        pass

    def engine_evaluation(self):
        pass

    def run(self):
        self.model_metrics = self.model_evaluation()
        self.engine_metrics = self.engine_evaluation()
