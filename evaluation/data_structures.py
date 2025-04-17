import hashlib
import logging

from PIL import Image as PILImage

from utils import parse_date_from_filepath

class CustomImage:
    """
    Custom image object that gathers data about each image : bytes, annotations, origin session
    """
    def __init__(self, image_path, session_id, timedelta, label):
        self.image_path = image_path
        self.session_id = session_id
        self.timestamp = parse_date_from_filepath(image_path)
        self.timedelta = timedelta
        self.label = label
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

class Session:
    """
    Objects that contains a list of images from a single session
    """
    def __init__(self, session_id: str, images: list[CustomImage] = []):
        self.session_id = session_id
        self.session_start = parse_date_from_filepath(session_id)
        self.images = images

    @property
    def label(self):
        """
        Define label as property as it needs to be recomputed for each image added or removed
        """
        return any(image.label for image in self.images)

    def get_session_label(self):
        image_labels = [image.label for image in self.images]
        return any(image_labels)

    def add_image(self, image_path, session_id, timedelta, label):
        self.images.append(CustomImage(image_path, session_id, timedelta, label))

    def __len__(self):
        return len(self.images)

