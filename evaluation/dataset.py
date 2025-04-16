import glob
import hashlib
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder
from PIL import Image as PILImage

from utils import parse_date_from_filepath, is_image, has_image_extension


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
        if self._image is None:
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
        self.label = self.get_session_label()

    def get_session_label(self):
        image_labels = [image.label for image in self.images]
        return any(image_labels)

    def add_image(self, image_path, session_id, timedelta, label):
        self.images.append(CustomImage(image_path, session_id, timedelta, label))

    def __len__(self):
        return len(self.images)


class EvaluationDataset:
    """
    Class that contains a dataset and some metadata about it. 
    It can be instantiated either with a local image folder or a hugging face repo
    """
    def __init__(self, datapath, save=False, dataset_ID=f"dataset_{datetime.now()}"):

        self.datapath = datapath
        self.save = save
        self.dataframe_ID = dataset_ID
        self.sessions: dict[str, Session] = {}

        self.is_local: bool = os.path.exists(self.datapath) # False if datapath is a HF repo
        self.dataframe : pd.DataFrame = self.init_from_folder() if self.is_local else self.init_from_hugging_face()

    def init_from_hugging_face(self, split="all"):
        """
        Builds a dataset dataframe from a huggingface repository.
        The repository is private so it requires authentification, use "huggingface-cli login" to authenticate.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Error : no Hugging Face found. Please authenticate with `huggingface-cli login`.")

        api = HfApi()

        dataset_info = api.dataset_info(self.datapath, token=token)
        if not dataset_info:
            raise ValueError(f"Error : {self.datapath} doesn't exist or is not accessible.")

        hf_dataset = load_dataset(self.datapath, split=split, trust_remote_code=True)
        
        # Retrieve images, annotations and dates as lists
        image_list = [element["image"] for element in hf_dataset]
        annotations = [element["annotations"] for element in hf_dataset]
        dates = [element["date"] for element in hf_dataset]

        # Identify common session and store data in a dataframe
        self.dataframe = self.determine_sessions(image_list, annotations, dates)


    def init_from_folder(self):
        """
        In order to init a dataset from a folder, the datapath must point to a folder containing two subfolders : images and labels
        images contains images, and labels contains annotations files names similarly as images with a .txt extension
        """

        def load_annotation(image_path):
            annotation_file = image_path.replace(".jpg", ".txt")
            if not os.path.isfile(annotation_file):
                annotations = ""
            else:
                with open(annotation_file, 'r') as file:
                    annotations = file.read()
            return annotations

        image_list = [image for image in glob.glob(f"{self.datapath}/*") if is_image(image)]
        annotations = [load_annotation(image_path) for image_path in image_list]
        dates = [parse_date_from_filepath(image_path) for image_path in image_list]

        # Identify common session and store data in a dataframe
        self.dataframe = self.determine_sessions(image_list, annotations, dates)

        if self.save:
            # Save the dataframe in a csv file
            output_csv = os.path.join(os.path.dirname(self.datapath), f"{os.path.basename(self.datapath)}.csv")
            self.dataframe.to_csv(output_csv, index=False)
            logging.info(f"DataFrame saved in {output_csv}")

    def build_dataset(self):
        """
        Create Session and CustomImage objects from dataset dataframe
        """
        for session in set(self.dataframe["session"]):
            session_df = self.dataframe[self.dataframe["session"] == session]
            for images in session_df:
                images = [CustomImage()]
                self.sessions.update({session : Session(session, images)})

    def determine_sessions(self, image_list, annotations, timestamps=None, max_delta=30):
        '''
        Parse images to detect files belonging to the same session by comparing camera name and capture dates.
        Expects file named as *_year_month_daythour_*
        '''
        image_list.sort()

        data = []
        current_session = None
        session_images = []

        for image_path, timestamp, annotation in zip(image_list, annotations, timestamps):            
            if not has_image_extension(image_path) or not timestamp:
                continue

            if not current_session:
                current_session = os.path.splitext(os.path.basename(image_path))[0]
                session_images = [image_path]
                session_start = timestamp
            else:
                last_image_timestamp = parse_date_from_filepath(session_images[-1])
                if (timestamp - last_image_timestamp) <= timedelta(minutes=max_delta):
                    session_images.append(image_path)
                else:
                    # More than 30 min between two captures -> Save current session and start a new one
                    for img in session_images:
                        data.append({
                            'image': img,
                            'session': current_session,
                            'label': annotation,
                            'delta': timestamp - session_start
                        })
                    current_session = os.path.splitext(os.path.basename(image_path))[0]
                    session_images = [image_path]

        # Save last session
        if session_images:
            for img in session_images:
                data.append({
                    'image': img,
                    'session': current_session,
                    'label': annotation,
                    'delta': timestamp - session_start
                })

        # Store everything in a csv
        df = pd.DataFrame(data)
        return df

    def get_images_from_session(self, session_id):
        """
        Return all images that were captured a the same time
        """
        session_df = self.dataframe[self.dataframe["session"] == session_id]
        return session_df["image"].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        """
        Allows to do: for image in dataset: ...
        """
        return iter(self.get_all_images())

    def get_all_images(self):
        """
        Returns a list of all images in the dataset
        """
        all_images = []
        for _, imgs in self.sessions.items():
            all_images += imgs
        return all_images
   
    def _group_by_session(self, images: list[CustomImage]):
        session_dict = {}
        for img in images:
            session_dict.setdefault(img.session_id, []).append(img)
        self.sessions = {sid: Session(sid, imgs) for sid, imgs in session_dict.items()}


