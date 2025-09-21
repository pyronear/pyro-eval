import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .dataset import EvaluationDataset
from .model import Model
from .path_manager import get_prediction_path, roc_image_path
from .prediction_manager import PredictionManager
from .utils import compute_metrics, find_matches, timing


class ModelEvaluator:
    def __init__(
        self,
        dataset: EvaluationDataset,
        model: Model,
        prediction_manager: PredictionManager,
        config: dict = {},
    ):
        self.dataset = dataset
        self.model = model
        self.prediction_manager = prediction_manager
        self.config = config["model"]
        self.iou_threshold = self.config["iou"]

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

        # Track image prediction status for further analysis
        self.predictions = {
            "tp": [],
            "tn": [],
            "fp": [],
            "fn": [],
        }

        self.prediction_file = get_prediction_path(self.config["model_path"])

    def track_predictions(self, image_path: str, fp: int, tp: int, fn: int) -> None:
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

    @timing("Model evaluation")
    def evaluate(self):
        """
        Compares predictions and labels to evaluate the model performance on the dataset
        """
        self.prediction_manager.predict(self.images)
        self.prediction_manager.save_predictions()

        nb_fp, nb_tp, nb_fn = 0, 0, 0

        y_true = []
        y_scores = []

        for image in self.images:
            # Labels
            gt_boxes = image.boxes_xyxy
            has_gt = len(gt_boxes) > 0
            y_true.append(int(has_gt))
            # Predictions - last element of the array is the confidence
            pred_boxes = np.array([pred[:-1] for pred in image.prediction])
            pred_scores = np.array(
                [pred[-1] for pred in image.prediction]
            )  # confidences

            # For ROC: take the highest confidence, or 0 if no prediction
            if len(pred_scores) > 0:
                y_scores.append(float(np.max(pred_scores)))
            else:
                y_scores.append(0.0)

            img_fp, img_tp, img_fn = find_matches(
                gt_boxes, pred_boxes, self.iou_threshold
            )
            self.track_predictions(image.path, img_fp, img_tp, img_fn)

            nb_fp += img_fp
            nb_tp += img_tp
            nb_fn += img_fn

        metrics = compute_metrics(
            false_positives=nb_fp, true_positives=nb_tp, false_negatives=nb_fn
        )
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        # Replace np.inf by a high value to be able to dump in a json
        max_threshold = 1e10
        thresholds = [
            max_threshold if np.isinf(val) else np.round(val, 5) for val in thresholds
        ]

        self.roc_data = {
            "fpr": np.round(fpr, 5).tolist(),
            "tpr": np.round(tpr, 5).tolist(),
            "thresholds": np.round(thresholds, 5).tolist(),
            "auc": auc_score,
        }

        self.save_roc_curve()

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fp": int(nb_fp),
            "tp": int(nb_tp),
            "fn": int(nb_fn),
            "tn": len(self.predictions["tn"]),
            "predictions": self.predictions,
            "roc_curve": self.roc_data,
        }

    def save_roc_curve(self) -> None:
        """
        Plots and saves roc curve
        """
        fpr = self.roc_data["fpr"]
        tpr = self.roc_data["tpr"]
        auc = self.roc_data["auc"]

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(roc_image_path(self.model.hash))
        plt.close()
