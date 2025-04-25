import glob
import hashlib
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder

from data_structures import CustomImage, Sequence
from utils import parse_date_from_filepath, is_image, has_image_extension

class EvaluationDataset:
    """
    Class that contains a dataset and metadata. 
    It can be instantiated either with a local image folder or a hugging face repo
    """
    def __init__(self, datapath, save=False, dataset_ID=None):

        self.datapath = datapath
        self.save = save
        self.sequences: list[Sequence] = []
        self.is_local: bool = os.path.exists(self.datapath) # False if datapath is a HF repo

        if not self.datapath:
            raise ValueError("No datapath provided to instanciate EvaluationDataset.")

        # Retrieve data from a local directory or a huggingface repository
        self.dataframe = self.init_from_folder() if self.is_local else self.init_from_hugging_face()

        # Build dataset from Sequence and CustomImage objects
        self.build_dataset()
        self.hash = self.compute_hash()
        self.dataset_ID = dataset_ID if dataset_ID else f"dataset_{datetime.now()}_{self.hash}"
        
        # Check that all image hashes are unique in the dataset
        self.check_unique_hashes()

    def init_from_hugging_face(self, split="all"):
        """
        Builds a dataset dataframe from a huggingface repository.
        The repository is private so it requires authentification, use "huggingface-cli login" to authenticate.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Error : no Hugging Face found. Please authenticate with `huggingface-cli login`.")

        api = HfApi()

        # Remove the first part of the url
        dataset_id = self.datapath.split("/datasets/")[-1]
        dataset_info = api.dataset_info(dataset_id, token=token)
        if not dataset_info:
            raise ValueError(f"Error : {self.datapath} doesn't exist or is not accessible.")

        # TODO : this needs memory management and optimization
        hf_dataset = load_dataset(dataset_id, split=split, trust_remote_code=True)
        
        # Retrieve images, annotations and dates as lists
        image_list = [element["image"] for element in hf_dataset]
        annotations = [element["annotations"] for element in hf_dataset]
        dates = [element["date"] for element in hf_dataset]

        # Identify common sequence and store data in a dataframe
        return self.determine_sequences(image_list, annotations, dates)

    def init_from_folder(self):
        """
        In order to init a dataset from a folder, the datapath must point to a folder containing two subfolders : images and labels
        images contains images, and labels contains annotations files names similarly as images with a .txt extension
        """
        if not os.path.isdir(self.datapath):
            raise FileNotFoundError(f"{self.datapath} is not a directory.")
    
        def load_annotation(image_path):
            """
            Loads boxes coordinnates from a txt file.
            """
            annotation_file = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            if not os.path.isfile(annotation_file):
                boxes = []
            else:
                with open(annotation_file, 'r') as file:
                    boxes = file.read().split("\n")
                    boxes = [box for box in boxes if len(box) > 0]

            return boxes

        image_list = [image for image in sorted(glob.glob(f"{self.datapath}/images/*")) if is_image(image)]
        annotations = [load_annotation(image_path) for image_path in image_list]
        timestamps = [parse_date_from_filepath(image_path)["date"] for image_path in image_list]

        # Identify common sequence and store data in a dataframe
        dataframe = self.determine_sequences(image_list, annotations, timestamps)

        if self.save:
            # Save the dataframe in a csv file
            output_csv = os.path.join(self.datapath, f"{os.path.basename(self.datapath)}.csv")
            dataframe.to_csv(output_csv, index=False)
            logging.info(f"DataFrame saved in {output_csv}")

        return dataframe

    def build_dataset(self):
        """
        Create Sequence and CustomImage objects from dataset dataframe
        Each Sequence contains a list of CustomImage, the dataset contains a dict with all sequences
        """

        for sequence_id, sequence_df in self.dataframe.groupby("sequence_id"):
            custom_images = [
                CustomImage(
                    image_path=row['image'],
                    sequence_id=sequence_id,
                    timedelta=row['delta'],
                    label=row['label']
                )
                for _, row in sequence_df.iterrows()
            ]

            self.sequences.append(Sequence(sequence_id, images=custom_images))

    def determine_sequences(self, image_list, annotations, timestamps=None, max_delta=30):
        '''
        Parse images to detect files belonging to the same sequence by comparing camera name and capture dates.
        Expects file named as *_year_month_daythour_*
        '''

        data = []
        current_sequence = None

        for image_path, annotation, timestamp in zip(image_list, annotations, timestamps):
            if not has_image_extension(image_path) or not timestamp:
                logging.info(f"Skipping {image_path} : wrong extenison or unable to retrieve timestamp.")
                continue

            image_prefix = os.path.basename(os.path.splitext(image_path)[0]).replace(timestamp.strftime("%Y_%m_%dt%H_%M_%S"), "")

            if not current_sequence:
                current_sequence = os.path.splitext(os.path.basename(image_path))[0]
                sequence_images = [(image_path, timestamp, annotation)]
                sequence_start = timestamp
                previous_image_timestamp = timestamp
                previous_prefix = image_prefix
            else:
                if (timestamp - previous_image_timestamp) <= timedelta(minutes=max_delta) and image_prefix == previous_prefix:
                    sequence_images.append((image_path, timestamp, annotation))
                else:
                    # More than 30 min between two captures -> Save current sequence and start a new one
                    for im_path, im_timestamp, im_label in sequence_images:
                        data.append({
                            'image': im_path,
                            'sequence_id': current_sequence,
                            'label': im_label,
                            'delta': im_timestamp - sequence_start
                        })

                    current_sequence = os.path.splitext(os.path.basename(image_path))[0]
                    sequence_images = [(image_path, timestamp, annotation)]
                    sequence_start = timestamp
                previous_image_timestamp = timestamp
                previous_prefix = image_prefix
        # Save last sequence
        if sequence_images:
            for image_path, timestamp, annotation in sequence_images:
                data.append({
                    'image': image_path,
                    'sequence_id': current_sequence,
                    'label': annotation,
                    'delta': timestamp - sequence_start
                })

        df = pd.DataFrame(data)
        return df

    def get_images_from_sequence(self, sequence_id):
        """
        Return all images that were captured a the same time
        """
        sequence_df = self.dataframe[self.dataframe["sequence_id"] == sequence_id]
        return sequence_df["image"].tolist()

    def get_all_images(self):
        """
        Returns a list of all images in the dataset
        """
        all_images = []
        for sequence in self.sequences:
            all_images.extend(sequence.images)
        return all_images


    def compute_hash(self):
        """
        Compute datashet hash based on the concatenation of each image hash.
        This can be used to detect dataset changes and provide identifiers
        # TODO : add
        """
        hashes = [img.hash for img in self.get_all_images()]
        combined = ''.join(hashes).encode('utf-8')
        return hashlib.sha256(combined).hexdigest()

    def check_unique_hashes(self) -> bool:
        """
        Check if all CustomImage instances have unique hashes.
        If duplicates are found, log the corresponding file paths.

        Returns:
            True if all hashes are unique, False otherwise.
        """
        hash_to_paths = defaultdict(list)

        # defaultdict(list) initialize the entry with {key : []} if key doesn't exist
        for img in self.get_all_images():
            hash_to_paths[img.hash].append(img.image_path)

        # Check for hash that have several path corresponding
        duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

        if duplicates:
            logging.warning("Duplicate image hashes detected:")
            for h, paths in duplicates.items():
                logging.warning(f"Hash {h} found in {len(paths)} files:")
                for path in paths:
                    logging.warning(f"  - {path}")
            return False

        return True

    def add_sequence(self, sequence: Sequence):
        self.sequences.append(sequence)

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        """
        Allows to do: for sequence in dataset: ...
        """
        return iter(self.sequences)

    def __repr__(self):
        
        nb_true_sequences = 0
        nb_true_images = 0
        for sequence in self.sequences:
            if sequence.label :
                nb_true_sequences += 1
            for image in sequence.images:
                if len(image.label) > 0:
                    nb_true_images += 1
        nb_images = len(self.get_all_images())
        repr_str = (
            f"CustomDataset with {len(self.sequences)} sequences and {nb_images} images.\n"
            f"Sequence Labels: {nb_true_sequences} True, {len(self.sequences) - nb_true_sequences} False\n"
            f"Image Labels: {nb_true_images} True, {nb_images - nb_true_images} False"
        )

        return repr_str
