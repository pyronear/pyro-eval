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
                predictions_pair = [
                    self.get_predictions(run, image_name=image_name, source=source)
                    for run in self.runs
                ]
                save_path = Path(out_path) / query / new_name
                # Apply bbox, concatenate and save
                self.process_image_pair(
                    image_pair=image_pair,
                    predictions_pair=predictions_pair,
                    save_path=save_path,
                )
                # except Exception as e:
                #     logging.error(f"Error processing {name} - {e}")

            elif source == "engine":
                os.makedirs(Path(out_path) / query / new_name, exist_ok=True)
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
                        except:
                            logging.error(f"Engine comparison : Error loading image: {image_name}")
                            continue
                        try:
                            predictions_pair = [
                                self.get_predictions(run, image_name=image_name, source=source)
                                for run in self.runs
                            ]
                        except:
                            logging.error(f"Engine comparison : Error loading predictions: {image_name}")
                            continue
                        final_image_name = f"run-A-{status_A}_run-B-{status_B}_{image_name}"
                        save_path = Path(out_path) / query / new_name / final_image_name

                        # Apply bbox, concatenate and save
                        self.process_image_pair(
                            image_pair=image_pair,
                            predictions_pair=predictions_pair,
                            save_path=save_path,
                        )
                        # except:
                        #     logging.error(f"Error processing {name}")

    def process_image_pair(
        self,
        image_pair: List[Image.Image],
        predictions_pair: List[str],
        save_path: str,
    ):
        bbox_images = [self.apply_bbox(im, pred) for im, pred in zip(image_pair, predictions_pair)]
        # Concatenate both in a single image
        final_image = self.concatenate_images(bbox_images[0], bbox_images[1])
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
        try:
            images = [
                Path(engine_path) / image_rel_path 
                for image_rel_path in self.trees[run.run_id]["engine"][sequence_name]
            ]
        except:
            print("not found")
            images = []

        return images

    def get_predictions(
            self,
            run: RunData,
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
            conf = random.uniform(0, 1)
            x1 = random.uniform(0, 1)
            y1 = random.uniform(0, 1)
            x2 = random.uniform(x1, 1)
            y2 = random.uniform(y1, 1)
            bboxes.append([x1, y1, x2, y2, conf])
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
            model_input_size : Tuple = (1024, 1024) # TODO : retrieve default value from CustomImage
        ) -> Image:
        """
        Add bbox to an image
        Bbox coordinates in xyxyn format are stored in the prediciton array
        Predictions on a resized version (1024, 1024), so we need to resize the image before adding them
        """
        w_orig, h_orig = im.size
        w_model, h_model = (model_input_size)

        # Compute scale to get the right coodrinate
        scale = min(w_orig / w_model, h_orig / h_model)

        im_drawn = im.copy()
        draw = ImageDraw.Draw(im_drawn)
        for pred in predictions:
            x1n, y1n, x2n, y2n, conf = pred[:5]

            # From xyxyn coordinates to coordinates in the 1024x1024 image
            x1, y1 = x1n * w_model, y1n * h_model
            x2, y2 = x2n * w_model, y2n * h_model

            # The scale to coordinates of the original image
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 3, y1 + 3), f"{conf:.2f}", fill="red")

        return im_drawn

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
