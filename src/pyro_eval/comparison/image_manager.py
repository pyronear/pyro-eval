import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from run_data import RunData
from pyro_eval.dataset import EvaluationDataset


class ImageManager:
    """Class that manages images in the comparison"""
    
    def __init__(self, runs: List[RunData]):
        self.runs = runs
        self.run_ids = [run.run_id for run in self.runs]
        self.image_dirs = {
            run.run_id : {
                "engine" : Path(run.engine_datapath),
                "model" : Path(run.model_datapath),
            }
            for run in runs
        }
        self.trees = {run_id : {} for run_id in self.run_ids}
        
        for run in self.runs:
            self.trees[run.run_id] = {
                # Recreate an instance of EvaluationDataset as the tree info is a late addition to metrics.json
                "model" : run.dataset.get("model", {}).get("tree_info") or EvaluationDataset(run.model_datapath).tree_info(),
                "engine" : run.dataset.get("engine", {}).get("tree_info") or EvaluationDataset(run.engine_datapath).tree_info(),
            }
        self.logger = logging.getLogger(__name__)
        
    def create_image_folder(
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
        └── outpath
            ├── model_query
            │   ├── run-A-TP_run-B-FN_image1.jpg
            │   ├── run-A-TP_run-B-TP_image2.jpg
            │   ├── run-A-FN_run-B-TP_image3.jpg
        Or for sequences : 
        ── outpath
            ├── engine_query
            │   ├── run-A-TP_run-B-FN_sequence1
            │   │   ├── image1.jpg
            │   │   ├── image2.jpg
            │   ├── run-A-TP_run-B-TP_sequence2
            │   │   ├── image4.jpg
            │   │   ├── image5.jpg
            │   │   ├── image6.jpg
        """
        if df.empty:
            self.logger.warning(f"Empty dataframe for the following query :  '{query}'")
            return
        os.makedirs(Path(out_path) / query, exist_ok=True)
        for _, row in df.iterrows():
            name = row["Name"] # name of the image or sequence
            status_A = row[self.run_ids[0]]
            status_B = row[self.run_ids[1]]
            new_name = f"run-A-{status_A}_run-B-{status_B}_{name}"
            if source == "model":
                # try:
                # Load original images
                image_pair = [Image.open(self.get_image_path(name, source, run)) for run in self.runs]
                # Apply detection bbox on each
                predictions = self.get_predictions(image_name=image_name, source=source)
                save_path = Path(out_path) / query / new_name
                # Apply bbox, concatenate and save
                self.process_image_pair(
                    image_pair=image_pair,
                    predictions=predictions,
                    save_path=save_path,
                )
                # except Exception as e:
                #     logging.error(f"Error processing {name} - {e}")

            elif source == "engine":
                os.makedirs(Path(out_path) / query / name, exist_ok=True)
                images_path = {
                    run.run_id : self.get_sequence_images(run, name)
                    for run in self.runs
                }
                for image_path in images_path[self.run_ids[0]]:
                    # Check that we find one image we the same name in the other dataset
                    if os.path.basename(image_path) in [os.path.basename(path) for path in images_path[self.run_ids[1]]]:
                        try:
                            image_name = os.path.basename(image_path)
                            image_pair = [Image.open(image_path) for _ in self.runs]
                            predictions = self.get_predictions(image_name=image_name, source=source)
                            final_image_name = f"run-A-{status_A}_run-B-{status_B}_{image_name}"
                            save_path = Path(out_path) / query / new_name / final_image_name

                            # Apply bbox, concatenate and save
                            self.process_image_pair(
                                image_pair=image_pair,
                                predictions=predictions,
                                save_path=save_path,
                            )
                        except:
                            logging.error(f"Error processing {name}")

    def process_image_pair(
        self,
        image_pair: List[Image.Image],
        predictions: List[str],
        save_path: str,
    ):
        bbox_images = [self.apply_bbox(im, predictions=predictions) for im in image_pair]
        # Concatenate both in a single image
        final_image = self.concatenate_images(bbox_images)
        final_image.save(save_path)

    def get_image_path(
        self,
        image_name: str, 
        source: str,
        run: RunData,
    ) -> Path :
        """
        Reconstruct image path from datasets info
        """
        root_path = self.image_dirs[run.run_id][source]
        tree = self.trees[run.run_id][source]
        relative_image_path = [path for seq in tree for path in tree[seq] if image_name in path][0]
        return Path(root_path) / relative_image_path

    def get_sequence_images(
            self,
            run : RunData,
            sequence_name : str,
        ) -> List[str]:
        """
        Retrieve all image paths for a given sequence
        """
        engine_path = run.engine_datapath
        images = [
            Path(engine_path) / image_rel_path 
            for image_rel_path in self.trees[run.run_id]["engine"][sequence_name]
        ]

        return images

    def get_predictions(
            self,
            image_name: str,
            source: str,
    ) -> np.ndarray:
        """
        TODO : retrieve actual predictions
        """
        
        image_size = (1024, 1024)
        num_bboxes = random.randint(0, 4)
        bboxes = []

        for _ in range(num_bboxes):
            x1 = random.randint(0, image_size[0] - 1)
            y1 = random.randint(0, image_size[1] - 1)
            x2 = random.randint(x1, image_size[0] - 1)
            y2 = random.randint(y1, image_size[1] - 1)
            bboxes.append((x1, y1, x2, y2))
        return  bboxes

    def concatenate_images(
            self,
            im1: Image,
            im2: Image
        ) -> Image:
        """
        Takes two image and concatenate them next to each other
        """
        w1, h1 = im1.size
        w2, h2 = im2.size
        max_height = max(h1, h2)
        if h1 < max_height:
            im1 = im1.resize((int(w1 * max_height / h1), max_height))

        if h2 < max_height:
            im2 = im2.resize((int(w2 * max_height / h2), max_height))

        new_image = Image.new('RGB', (im1.width + im2.width, max_height))

        new_image.paste(im1, (0, 0))
        new_image.paste(im2, (im1.width, 0))
        return new_image

    def apply_bbox(
            self,
            im : Image,
            predictions : np.array,
            target_size : Tuple = (1024, 1024) # TODO : retrieve default value from CustomImage
        ) -> Image:
        """
        Add bbox to an image
        Bbox coordinates in xyxyn format are stored in the prediciton array
        Predictions on a resized version (1024, 1024), so we need to resize the image before adding them
        """
        w, h = im.size
        w_t, h_t = target_size
        resized_im = im.resize(target_size)
        draw = ImageDraw.Draw(resized_im)

        for pred in predictions:
            x1, y1, x2, y2 = pred[:4]
            conf = pred[4]

            # Convert normalized to absolute coords
            x1_abs, y1_abs = int(x1 * w_t), int(y1 * h_t)
            x2_abs, y2_abs = int(x2 * w_t), int(y2 * h_t)

            # Draw rectangle
            draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], outline="red", width=2)

            # Add label
            label = f"{conf:.2f}"
            draw.text((x1_abs + 3, y1_abs + 3), label, fill="red")

        im = resized_im.resize((w, h))
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
