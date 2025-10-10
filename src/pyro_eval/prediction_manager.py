import json
import logging
from typing import Dict, List

import numpy as np
from pyroengine.vision import Classifier

from .data_structures import CustomImage
from .model import Model
from .utils import export_model, make_dict_json_compatible

class PredictionManager:
    def __init__(
        self,
        model: Model,
        prediction_file: str,
        config: Dict,
        use_existing_predictions: bool = True,
    ):
        self.model = model
        self.prediction_file = prediction_file
        self.config = config
        self.use_existing_predictions = use_existing_predictions
        self.predictions = (
            self.load_predictions() if self.use_existing_predictions else {}
        )

        # Instanciate a Classifier to use the post processing method
        model_path = config["model_path"]

        if model_path.endswith(".pt"):
            model_path = export_model(model_path) # Export to onnx format

        self.dummy_model = Classifier(
            model_path = model_path,
            conf = config["model"]["conf"],
            imgsz = config["model"]["imgsz"],
            iou = config["model"]["iou"],
        )

    def load_predictions(self) -> Dict[str, np.ndarray]:
        """
        Loads predictions from a json file
        """
        loaded_predictions = {}
        try:
            with open(self.prediction_file, "r") as f:
                data = json.load(f)

            for image_name, prediction in data.items():
                loaded_predictions[image_name] = np.array(prediction)

            return loaded_predictions

        except Exception as e:
            logging.error(
                f"Failed to load predictions from {self.prediction_file}: {e}"
            )
            return {}

    def save_predictions(self):
        # Load existing predictions to make sure we save everything
        loaded_predictions = self.load_predictions()
        new_predictions = self.predictions.copy()
        new_predictions.update(loaded_predictions)
        try:
            with open(self.prediction_file, "w") as f:
                json.dump(make_dict_json_compatible(new_predictions), f)
        except:
            logging.error(f"Unable to save predictions in {self.prediction_file}")

    def predict(self, images: List[CustomImage]):
        """
        Run predictions on an image list
        If use_existing_predictions is False, existing predictions are not loaded and everything is recomputed
        If use_existing_predictions is True, predictions are loaded from a json and only the missing ones are computed
        """

        for image in images:
            if image.name not in self.predictions:
                image.prediction = self.model.inference(image)
                self.predictions[image.name] = image.prediction
            else:
                image.prediction = self.predictions[image.name]

    def engine_post_process(
            self,
            preds: np.ndarray,
        ) -> np.ndarray:
        """
        This method post processes prediction with engine filtering:
        Remove bboxes with low confidence, remove larger bboxes, apply nms
        Preds should have onnx format 
        """
        # Drop low-confidence predictions and apply nms
        preds = self.dummy_model.post_process(preds, pad=(0, 0))

        # Filter predictions larger than max_bbox_size
        preds = np.clip(preds, 0, 1)
        preds = preds[(preds[:, 2] - preds[:, 0]) < self.config["engine"]["max_bbox_size"], :]
        preds = np.reshape(preds, (-1, 5))

        return preds