import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from utils import parse_date_from_filepath, xywh2xyxy

@dataclass
class CustomImage:
    """
    Dataclass for a custom image object that gathers data about each image : bytes, annotations, origin sequence
    """
    image_path: str
    sequence_id: str
    timedelta: float
    label: str

    timestamp: str = field(init=False)
    hash: str = field(init=False)
    prediction: Optional[str] = field(default=None)

    def __post_init__(self):
        self.timestamp = parse_date_from_filepath(self.image_path)["date"]
        self.hash = self.compute_hash()
    
    def load(self) -> PILImage.Image:
        """
        Load image only when needed
        """
        try:
            image = PILImage.open(self.image_path)
        except:
            image = None
            logging.error(f"Unable to load image : {self.image_path}")
        return image

    def compute_hash(self):
        hash_md5 = hashlib.md5()
        with open(self.image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def label_xyxy(self):
        """
        Returns a numpy array containing bounding boxes coordinates in xyxy format.
        """
        # Remove trailing \n, whitespaces, first value of the predicted array (class id) for each box
        boxes = [np.array(box.strip().split(" ")[1:5]).astype(float) for box in self.label]
        # Translate into xyxy coordinates and return
        return [xywh2xyxy(box) for box in boxes]

class Sequence:
    """
    Objects that contains a list of images from a single sequence
    """
    def __init__(self, sequence_id: str, images: list[CustomImage] = []):
        self.sequence_id = sequence_id
        self.images = images
        self.sequence_start = self.images[0].timestamp

    @property
    def label(self):
        """
        Define label as property as it needs to be recomputed for each image added or removed
        """
        return any(image.label for image in self.images)

    def get_sequence_label(self):
        image_labels = [image.label for image in self.images]
        return any(image_labels)

    def add_image(self, image_path, sequence_id, timedelta, label):
        self.images.append(CustomImage(image_path, sequence_id, timedelta, label))

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        """
        for image in sequence: will iterate over CustomImages in self.images
        """
        return iter(self.images)
