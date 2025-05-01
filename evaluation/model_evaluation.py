from dataset import EvaluationDataset
from model import Model
from utils import compute_metrics, find_matches

class ModelEvaluator:
    def __init__(self, dataset: EvaluationDataset, config={}, device=None):
        self.dataset = dataset
        self.config = config
        self.model_path = self.config.get("model_path", None)
        self.inference_params = self.config.get("inference_params", {})

        # Load model
        self.model = Model(self.model_path, self.inference_params, device)

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

    def run_predictions(self):
        # Run pred for each CustomImage in the EvaluationDataset
        images = self.images
        for image in images:
            image.prediction = self.model.inference(image)

    def evaluate(self):
        """
        Compares predictions and labels to evaluate the model performance on the dataset
        """
        self.run_predictions()

        nb_fp, nb_tp, nb_fn = 0, 0, 0
        for image  in self.images:
            # Labels
            gt_boxes = image.label_xyxy
            # Predictions
            pred_boxes = image.prediction
            fp, tp, fn = find_matches(gt_boxes, pred_boxes, self.config.get("iou", 0.1))

            nb_fp += fp
            nb_tp += tp
            nb_fn += fn

        metrics = compute_metrics(false_positives=fp, true_positives=tp, false_negatives=fn)

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "fp" : int(nb_fp),
            "tp" : int(nb_tp),
            "fn" : int(nb_fn),
            }