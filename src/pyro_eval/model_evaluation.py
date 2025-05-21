from .dataset import EvaluationDataset
from .model import Model
from .utils import compute_metrics, find_matches


class ModelEvaluator:
    def __init__(self, dataset: EvaluationDataset, config={}, device=None):
        self.dataset = dataset
        self.config = config
        self.model_path = self.config.get("model_path", None)
        self.inference_params = self.config.get("inference_params", {})
        self.iou_threshold = self.config.get("iou", 0.1)

        # Load model
        self.model = Model(self.model_path, self.inference_params, device)

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

        # Track image prediction status for further analysis
        self.predictions = {
            "tp": [],
            "tn": [],
            "fp": [],
            "fn": [],
        }

    def run_predictions(self):
        # Run pred for each CustomImage in the EvaluationDataset
        for image in self.images:
            image.prediction = self.model.inference(image)

    def track_predictions(self, fp, tp, fn, image_path):
        """
        Track and stroe predictions for each image
        """
        if fp > 0:
            self.predictions["fp"].append(image_path)
        elif tp > 0:
            self.predictions["tp"].append(image_path)
        if fn > 0:
            self.predictions["fn"].append(image_path)
        else:
            self.predictions["tn"].append(image_path)

    def evaluate(self):
        """
        Compares predictions and labels to evaluate the model performance on the dataset
        """
        self.run_predictions()

        nb_fp, nb_tp, nb_fn = 0, 0, 0

        for image in self.images:
            # Labels
            gt_boxes = image.boxes_xyxy
            # Predictions
            pred_boxes = image.prediction
            fp, tp, fn = find_matches(gt_boxes, pred_boxes, self.iou_threshold)
            self.track_predictions(fp, tp, fn, image.image_path)

            nb_fp += fp
            nb_tp += tp
            nb_fn += fn
        metrics = compute_metrics(
            false_positives=nb_fp, true_positives=nb_tp, false_negatives=nb_fn
        )

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fp": int(nb_fp),
            "tp": int(nb_tp),
            "fn": int(nb_fn),
            "predictions": self.predictions,
        }
