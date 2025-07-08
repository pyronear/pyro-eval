import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from run_data import RunData

class ImageManager:
    """Class that manages images in the comparison"""
    
    def __init__(self, runs: List[RunData]):
        self.runs = runs
        self.run_ids = [run.run_id for run in self.runs]
        self.image_root_dirs = {
            run.run_id : {
                "engine" : Path(run.engine_datapath),
                "model" : Path(run.model_datapath),
            }
            for run in runs
        }

    def save_images(
            self,
            df: pd.DataFrame,
            out_path: str,
            source: str,
            query: str,
        ):
        """
        Saves images from a data frame:
        - Retrieve image and status
        - Apply detection bbox and concatenate both image into one
        - Save them in a folder named after the filtering query 
        """
        os.makedirs(Path(out_path) / query, exist_ok=True)
        for row in df:
            image_name = row["image_name"]
            status_A = row[self.run_ids[0]]
            status_B = row[self.run_ids[1]]
            new_name = f"run-A-{status_A}_run-B-{status_B}_{image_name}"
            # Load original images
            images = [Image.open(self.get_image_path(image_name, source, run)) for run in self.runs]
            # Apply detection bbox on each
            bbox_images = [self.apply_bbox(im) for im in images]

            # Concatenate both in a single image
            final_image = self.concatenate_images(bbox_images)
            final_image.save(Path(out_path) / query / new_name)

    def get_image_path(
        image_name: str, 
        source: str,
        run: RunData,
    ) -> Path :
        root_path = run.model_datapath if source == "model" else run.engine_datapath
        return Path(root_path) / image_name

    def concatenate_images(
            self,
            im1: Image,
            im2: Image
        ) -> Image:
        """
        Takes two image and concatenate them next to each other
        """
        pass 

    def apply_bbox(
            self,
            im : Image,
            prediction : np.array,
        ) -> Image:
        """
        Add bbox to an image
        bbox coordinates in xyxyn format are stored in the prediciton array
        """
        draw = ImageDraw.Draw(im)
        w, h = im.size

        for pred in prediction:
            x1, y1, x2, y2 = pred[:4]
            conf = pred[4]

            # Convert normalized to absolute coords
            x1_abs, y1_abs = int(x1 * w), int(y1 * h)
            x2_abs, y2_abs = int(x2 * w), int(y2 * h)

            # Draw rectangle
            draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], outline="red", width=2)

            # Add label
            label = f"{conf:.2f}"
            draw.text((x1_abs + 3, y1_abs + 3), label, fill="red")

        return im

    def display_label(
            self,
            im: Image,
            label: str,
        ) -> Image:
        """
        Adds the detection label TP/FP/TN/FN to the image
        """
        draw = ImageDraw.Draw(im)
        draw.text((5, 5), label, fill="red")
        return im

    def get_sequence_images(
            self,
            run_id : str,
            sequence_name : str,
        ) -> List[str]:
        """
        Retrieve all image paths for a given sequence
        """
        engine_path = self.image_root_dirs[run_id]["engine"]
        sequence_path = engine_path / sequence_name
        images = [
            im 
            for im in glob(f"{sequence_path}/*")
            if im.endswith(".jpg")
        ]
        return images