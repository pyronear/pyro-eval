import re
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from pandas import Timedelta

EXT = [".jpg", ".png", ".tif", ".jpeg", ".tiff"]

def is_image(image_path):
    return os.path.isfile(image_path) and has_image_extension(image_path)

def has_image_extension(image_path):
    return os.path.splitext(image_path)[-1].lower() in EXT

def parse_date_from_filepath(filepath):
    filename = os.path.basename(filepath)
    '''Extracts date from filename, typcally : pyronear_sdis-07_brison-200_2024-01-26t11-13-37.jpg'''
    pattern = r'_(\d{4})_(\d{2})_(\d{2})t(\d{2})_(\d{2})_(\d{2})\.(jpg|png)$'

    # Search for the pattern in the filename
    match = re.search(pattern, filename.lower())

    if match:
        # Extract components
        prefix = match.group(0)
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6))

        # Create datetime object
        file_datetime = datetime(year, month, day, hour, minute, second)
        return {
            "prefix": prefix, 
            "date": file_datetime,
        }

    return None

def make_dict_json_compatible(data):
    '''
    Replaces values to be able dump a dict in a json:
        - Replace True/False by "true"/"false"
        - Convert Timedelta to str
        - Convert int64 to int
    '''
    if isinstance(data, dict):
        return {key: make_dict_json_compatible(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_dict_json_compatible(item) for item in data]
    elif isinstance(data, np.bool_):
        return "True" if data else "False"
    elif isinstance(data, Timedelta):
        # Convertir Timedelta en chaîne de caractères
        return str(data)
    elif np.issubdtype(type(data), np.integer):
        # Convert int64 in native int
        return int(data)
    else:
        return data


def metrics_visualization(metrics, session_df):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image-level confusion matrix
    confusion_matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    cax = axes[0].matshow(confusion_matrix, cmap='Blues')
    axes[0].set_title('Image Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Annotate the confusion matrix
    for (i, j), val in np.ndenumerate(confusion_matrix):
        axes[0].text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.colorbar(cax, ax=axes[0])

    # Session detection delay histogram
    detection_delays = session_df[session_df['label'] & session_df['has_detection']]['detection_delay'].dt.total_seconds()
    axes[1].hist(detection_delays, bins=15, color='blue', alpha=0.7)
    axes[1].set_title('Detection Delay (Seconds)')
    axes[1].set_xlabel('Seconds since session start')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()