import hashlib
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder

from .data_structures import CustomImage, Sequence
from .utils import (
    EXTENSIONS,
    has_image_extension,
    is_image,
    parse_date_from_filepath,
    replace_extension,
)


class EvaluationDataset:
    """
    Class that contains a dataset and metadata.
    It can be instantiated either with a local image folder or a hugging face repo
    """

    def __init__(self, datapath, dataset_ID=None):
        self.datapath = datapath
        self.sequences: list[Sequence] = []
        self.is_local: bool = os.path.exists(
            self.datapath
        )  # False if datapath is a HF repo

        if not self.datapath:
            raise ValueError("No datapath provided to instanciate EvaluationDataset.")

        # Retrieve data from a local directory or a huggingface repository
        self.dataframe = (
            self.init_from_folder() if self.is_local else self.init_from_hugging_face()
        )

        # Build dataset from Sequence and CustomImage objects
        self.build_dataset()
        self.hash = self.compute_hash()
        self.dataset_ID = (
            dataset_ID if dataset_ID else f"dataset_{datetime.now()}_{self.hash[:10]}"
        )

        self.dataframe = self.get_sequence_label()

        # Check that all image hashes are unique in the dataset
        self.check_unique_hashes()

    def init_from_hugging_face(self, split="all"):
        """
        Builds a dataset dataframe from a huggingface repository.
        The repository is private so it requires authentification, use "huggingface-cli login" to authenticate.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "Error : no Hugging Face found. Please authenticate with `huggingface-cli login`."
            )

        api = HfApi()

        # Remove the first part of the url
        dataset_id = self.datapath.split("/datasets/")[-1]
        dataset_info = api.dataset_info(dataset_id, token=token)
        if not dataset_info:
            raise ValueError(
                f"Error : {self.datapath} doesn't exist or is not accessible."
            )

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
            # Get labels folder
            annotation_file = str(image_path).replace("/images/", "/labels/")
            # Change file extension to .txt
            annotation_file = replace_extension(annotation_file, EXTENSIONS, ".txt")

            if not os.path.isfile(annotation_file):
                boxes = []
            else:
                with open(annotation_file, "r") as file:
                    boxes = file.read().split("\n")
                    boxes = [box for box in boxes if len(box) > 0]

            return boxes

        image_filepaths = list((Path(self.datapath) / "images").glob("**/*.jpg"))
        logging.info(f"Found {len(image_filepaths)} images in dataset")
        image_list = [image for image in sorted(image_filepaths) if is_image(image)]
        annotations = [load_annotation(image_path) for image_path in image_list]
        timestamps = [
            parse_date_from_filepath(image_path).get("date", None)
            for image_path in image_list
        ]
        self.nb_invalid_images = timestamps.count(None)
        logging.info(f"No timestamp found on {self.nb_invalid_images} images.")

        # TODO : investigate storing the 3 lists above in an OrderedDict
        self.images_info = {
            image_path: {
                "prefix": parse_date_from_filepath(image_path).get("prefix", None),
                "date": parse_date_from_filepath(image_path).get("date", None),
            }
            for image_path in image_list
        }

        # Identify common sequence and store data in a dataframe
        dataframe = self.determine_sequences(image_list, annotations, timestamps)

        return dataframe

    def build_dataset(self):
        """
        Create Sequence and CustomImage objects from dataset dataframe
        Each Sequence contains a list of CustomImage, the dataset contains a dict with all sequences
        """

        for sequence_id, sequence_df in self.dataframe.groupby("sequence_id"):
            custom_images = [
                CustomImage(
                    path=row["image"],
                    sequence_id=sequence_id,
                    timedelta=row["delta"],
                    boxes=row["boxes"],
                )
                for _, row in sequence_df.iterrows()
            ]

            self.sequences.append(
                Sequence(images=custom_images, sequence_number=len(self.sequences))
            )

    def determine_sequences(self, image_list, annotations, timestamps, max_delta=30):
        """
        Parse images to detect files belonging to the same sequence by comparing camera name and capture dates.
        Expects file named as *_year_month_daythour_*
        """

        data = []
        current_sequence = None

        for image_path, boxes, timestamp in zip(image_list, annotations, timestamps):
            # TODO : Better handle images without timestamps
            if not has_image_extension(image_path) or not timestamp:
                logging.info(
                    f"Skipping {image_path} : wrong extension or unable to retrieve timestamp."
                )
                continue

            # TODO : Date format to standardize
            image_prefix = self.images_info.get(image_path, {}).get("prefix")

            if not current_sequence:
                current_sequence = os.path.splitext(os.path.basename(image_path))[0]
                sequence_images = [(image_path, timestamp, boxes)]
                sequence_start = timestamp
                previous_image_timestamp = timestamp
                previous_prefix = image_prefix
            else:
                if (timestamp - previous_image_timestamp) <= timedelta(
                    minutes=max_delta
                ) and image_prefix == previous_prefix:
                    sequence_images.append((image_path, timestamp, boxes))
                else:
                    # More than 30 min between two captures -> Save current sequence and start a new one
                    for im_path, im_timestamp, im_boxes in sequence_images:
                        data.append(
                            {
                                "image": im_path,
                                "sequence_id": current_sequence,
                                "boxes": im_boxes,
                                "delta": im_timestamp - sequence_start,
                            }
                        )

                    current_sequence = os.path.splitext(os.path.basename(image_path))[0]
                    sequence_images = [(image_path, timestamp, boxes)]
                    sequence_start = timestamp
                previous_image_timestamp = timestamp
                previous_prefix = image_prefix
        # Save last sequence
        if sequence_images:
            for image_path, timestamp, boxes in sequence_images:
                data.append(
                    {
                        "image": image_path,
                        "sequence_id": current_sequence,
                        "boxes": boxes,
                        "delta": timestamp - sequence_start,
                    }
                )

        df = pd.DataFrame(data)
        return df

    def get_sequence_label(self):
        """
        Add a column with sequence label to the existing dataframe
        """
        self.dataframe["sequence_label"] = [""] * len(self.dataframe)
        for sequence in self.sequences:
            for image in sequence.images:
                self.dataframe.loc[
                    self.dataframe["image"] == image.path, "sequence_label"
                ] = sequence.label
        return self.dataframe

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
        combined = "".join(hashes).encode("utf-8")
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
            hash_to_paths[img.hash].append(img.path)

        # Check for hash that have several path corresponding
        self.duplicates = {
            h: paths for h, paths in hash_to_paths.items() if len(paths) > 1
        }

        if self.duplicates:
            logging.warning(
                f"🔴 {len(self.duplicates.keys())} duplicate image hashes detected"
            )
            for h, paths in self.duplicates.items():
                logging.warning(f"Hash {h} found in {len(paths)} files:")
                for path in paths:
                    logging.warning(f"  - {path}")
            return False

        return True

    def add_sequence(self, sequence: Sequence):
        self.sequences.append(sequence)

    def dump(self, path=None):
        output_csv = (
            path
            if path
            else os.path.join(self.datapath, f"{os.path.basename(self.datapath)}.csv")
        )
        self.dataframe.to_csv(output_csv, index=False)
        logging.info(f"DataFrame saved in {output_csv}")

    def compute_dataset_statistics(self):
        """
        Computes stastistics on the dataset built
        """
        nb_true_sequences = 0
        nb_true_images = 0
        for sequence in self.sequences:
            if sequence.label:
                nb_true_sequences += 1
            for image in sequence.images:
                if image.label:
                    nb_true_images += 1
        nb_images = len(self.get_all_images())
        one_img_sequences = [seq for seq in self.sequences if len(seq) == 1]
        multi_img_sequences = [len(seq) for seq in self.sequences if len(seq) > 1]
        avg_nb_img_per_sequence = (
            len(self.dataframe) / len(self.sequences) if len(self.sequences) != 0 else 0
        )
        avg_nb_img_per_sequence_multi = (
            sum(multi_img_sequences) / len(multi_img_sequences)
            if len(multi_img_sequences) != 0
            else 0
        )

        return {
            "nb_images": nb_images,
            "nb_true_sequences": nb_true_sequences,
            "nb_true_images": nb_true_images,
            "avg_nb_img_per_sequence": avg_nb_img_per_sequence,
            "nb_one_img_sequences": len(one_img_sequences),
            "avg_nb_img_per_sequence_multi": avg_nb_img_per_sequence_multi,
        }

    def stat_plots(self):
        sequence_lengths = [len(sequence) for sequence in self.sequences]

        six_p = 0
        lengths_distribution = [0] * (max(sequence_lengths) + 1)
        for length in sequence_lengths:
            if length > 4:
                six_p += 1
            lengths_distribution[length] += 1
        plt.figure()
        plt.yscale("log")
        plt.hist(
            sequence_lengths,
            bins=max(sequence_lengths) + 1,
            range=(0, max(sequence_lengths) + 1),
            edgecolor="black",
        )
        plt.title("Sequence length distribution")
        plt.show()

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        """
        Allows to do: for sequence in dataset: ...
        """
        return iter(self.sequences)

    def __repr__(self):
        """
        Print dataset info
        """
        stats = self.compute_dataset_statistics()

        total_sequences = len(self.sequences)
        nb_images = stats.get("nb_images", "N/A")
        nb_true_sequences = stats.get("nb_true_sequences", "N/A")
        nb_false_sequences = (
            total_sequences - nb_true_sequences
            if isinstance(nb_true_sequences, int)
            else "N/A"
        )
        nb_true_images = stats.get("nb_true_images", "N/A")
        nb_false_images = (
            nb_images - nb_true_images
            if isinstance(nb_images, int) and isinstance(nb_true_images, int)
            else "N/A"
        )
        avg_per_seq = stats.get("avg_nb_img_per_sequence", "N/A")
        avg_per_seq_multi = stats.get("avg_nb_img_per_sequence_multi", "N/A")
        nb_one_img_sequences = stats.get("nb_one_img_sequences", "N/A")

        repr_str = (
            f"+{'-' * 48}+\n"
            f"| {'CustomDataset Summary':^46} |\n"
            f"+{'=' * 48}+\n"
            f"| Total images                        | {nb_images:>7} |\n"
            f"| Total sequences                     | {total_sequences:>7} |\n"
            f"| Invalid images (invalid naming)     | {self.nb_invalid_images:>7} |\n"
            f"| Number of duplicates                | {len(self.duplicates.keys()):>7} |\n"
            f"+{'-' * 48}+\n"
            f"| Sequences with label=True           | {nb_true_sequences:>7} |\n"
            f"| Sequences with label=False          | {nb_false_sequences:>7} |\n"
            f"| Images with label=True              | {nb_true_images:>7} |\n"
            f"| Images with label=False             | {nb_false_images:>7} |\n"
            f"+{'-' * 48}+\n"
            f"| Avg images/sequence                 | {avg_per_seq:>7.1f} |\n"
            f"| Avg images/sequence (multi only)    | {avg_per_seq_multi:>7.1f} |\n"
            f"| Sequences with only 1 image         | {nb_one_img_sequences:>7} |\n"
            f"+{'-' * 48}+"
        )

        return repr_str
