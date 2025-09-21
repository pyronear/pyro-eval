import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage
from pyroengine.utils import letterbox

from .utils import parse_date_from_filepath, xywh2xyxy


@dataclass
class CustomImage:
    """
    Dataclass for a custom image object that gathers data about each image : bytes, annotations, origin sequence
    """

    path: str
    sequence_id: str
    timedelta: float
    boxes: List[str]

    timestamp: str = field(init=False)
    hash: str = field(init=False)
    prediction: Optional[str] = field(
        default=None
    )  # Formatted as a 5-array of predictions [[boxes.xyxyn, conf]]
    image_size = (1024, 1024)

    def __post_init__(self):
        self.timestamp = parse_date_from_filepath(self.path)["date"]
        self.hash = self.compute_hash()
        self.label: bool = len(self.boxes) > 0
        self.name: str = os.path.basename(self.path)

    def load(self, resize=False) -> PILImage.Image:
        """
        Load image only when needed
        """
        try:
            image = PILImage.open(self.path)
        except:
            image = None
            logging.error(f"Unable to load image : {self.path}")
        if resize:
            image, _ = letterbox(np.array(image), self.image_size)

        return image

    def compute_hash(self):
        hash_md5 = hashlib.md5()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def boxes_xyxy(self):
        """
        Returns a list of bounding boxes coordinates in xyxy format.
        """
        # Handle empty case safely
        if not self.boxes:
            return []
        try:
            # Remove trailing \n, whitespaces, first value of the predicted array (class id) for each box
            boxes = [
                np.array(box.strip().split(" ")[1:5]).astype(float)
                for box in self.boxes
            ]
            # Translate into xyxy coordinates and return
            return [xywh2xyxy(box) for box in boxes]
        except Exception as e:
            logging.warning(f"Failed to parse boxes for image {self.path}: {e}")
            return []

    @property
    def preds_onnx_format(self):
        """
        Convert predictions of shape (N, 5) in xyxyn format + confidence
        to a pseudo ONNX format to pass into the Engine:
            - shape (85, N) - transposed compared to ultralytics output
            - xywh not normalized + conf + placeholder for 80 classes (left empty)
        """

        preds = np.array(self.prediction)

        if preds.size == 0:
            return np.empty((5, 0))

        boxes_xyxyn = preds[:, :4]  # [x1, y1, x2, y2]
        confidences = preds[:, 4]  # [conf]

        # Convert from xyxy to xywh et denormalize
        x1, y1, x2, y2 = (
            boxes_xyxyn[:, 0],
            boxes_xyxyn[:, 1],
            boxes_xyxyn[:, 2],
            boxes_xyxyn[:, 3],
        )

        w, h = self.image_size
        x_center = (x1 + x2) / 2 * w
        y_center = (y1 + y2) / 2 * h
        width = (x2 - x1) * w
        height = (y2 - y1) * h

        # Build array with ONNX format : vertical stack (5, n_detections)
        onnx_predictions = np.vstack([x_center, y_center, width, height, confidences])

        return onnx_predictions

    def resize(self):
        """
        Resize to target size and apply letterbox algorithm
        """


class Sequence:
    """
    Objects that contains a list of images from a single sequence
    """

    def __init__(
        self,
        images: list[CustomImage] = [],
        sequence_number: int = None,
    ):
        self.images = images
        self.sequence_start = self.images[0].timestamp
        self.sequence_number = sequence_number
        self.id = self.get_sequence_id()

    @property
    def label(self):
        """
        Define label as property as it needs to be recomputed for each image added or removed
        """
        return any(image.label for image in self.images)

    def get_sequence_label(self):
        image_labels = [image.label for image in self.images]
        return any(image_labels)

    def get_sequence_id(self):
        """
        Retrieve information from the first image name to following the following naming convention:
        {date}_{origin}_{organization}_{camera}-{azimuth}-{sequence-number}
        A usual image name is force-06_cabanelle-327_2024-04-03T09-55-17.jpg
        If the information is not available, sequence id will be the raw name of the first image
        """
        parsing = parse_date_from_filepath(self.images[0].name)
        date = parsing["date"].strftime("%Y-%m-%dT%H-%M-%S")
        sequence_number = self.sequence_number or ""
        return date + parsing["prefix"] + str(sequence_number)

    def add_image(self, image_path, sequence_id, timedelta, label):
        self.images.append(CustomImage(image_path, sequence_id, timedelta, label))

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        """
        for image in sequence: will iterate over CustomImages in self.images
        """
        return iter(self.images)
