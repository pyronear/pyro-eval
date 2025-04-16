import glob
import os
from datetime import timedelta

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from utils import parse_date_from_filename, is_image

def determine_sessions(image_folder, annotation_folder, dataset_name):
    '''
    Parse images to detect files belonging to the same session by comparing camera name and capture dates.
    Works with wildfire2025 and DS_fp formats (expects file named as *_year_month_daythour_)
    '''
    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_files.sort()

    data = []
    current_session = None
    session_images = []

    for image_file in image_files:
        image_date = parse_date_from_filename(image_file)
        if not is_image(image_file) or not image_date:
            continue

        if not current_session:
            current_session = os.path.splitext(os.path.basename(image_file))[0]
            session_images = [image_file]
        else:
            last_image_date = parse_date_from_filename(session_images[-1])
            if (image_date - last_image_date) <= timedelta(minutes=30):
                session_images.append(image_file)
            else:
                # More than 30 min between two captures -> Save current session and start a new one
                for img in session_images:
                    annotation_file = os.path.join(annotation_folder, os.path.splitext(os.path.basename(img))[0] + '.txt')
                    if not os.path.isfile(annotation_file):
                        annotations = ""
                    else:
                        with open(annotation_file, 'r') as file:
                            annotations = file.read()
                    # TODO : add time delta from first image of the session
                    data.append({
                        'image': img,
                        'session': current_session,
                        'label': annotations,
                        'original_dataset': dataset_name
                    })
                current_session = os.path.splitext(os.path.basename(image_file))[0]
                session_images = [image_file]

    # Save last session
    if session_images:
        for img in session_images:
            annotation_file = os.path.join(annotation_folder, os.path.splitext(os.path.basename(img))[0] + '.txt')
            if not os.path.isfile(annotation_file):
                    annotations = ""
            else:
                with open(annotation_file, 'r') as file:
                    annotations = file.read()
            data.append({
                'image': img,
                'session': current_session,
                'label': annotations,
                'original_dataset': dataset_name
            })


    # Store everything in a csv
    df = pd.DataFrame(data)
    output_csv = os.path.join(os.path.dirname(image_folder), f"{os.path.basename(image_folder)}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Dataset dataframe saved in {output_csv}")