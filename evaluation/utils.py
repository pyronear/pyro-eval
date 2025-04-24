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

def metrics_visualization(metrics, sequence_df):
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

    # Sequence detection delay histogram
    detection_delays = sequence_df[sequence_df['label'] & sequence_df['has_detection']]['detection_delay'].dt.total_seconds()
    axes[1].hist(detection_delays, bins=15, color='blue', alpha=0.7)
    axes[1].set_title('Detection Delay (Seconds)')
    axes[1].set_xlabel('Seconds since sequence start')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def xywh2xyxy(x):
    """Function to convert bounding box format from center to top-left corner"""
    y = np.zeros_like(x)
    y[0] = x[0] - x[2] / 2  # x_min
    y[1] = x[1] - x[3] / 2  # y_min
    y[2] = x[0] + x[2] / 2  # x_max
    y[3] = x[1] + x[3] / 2  # y_max
    return y


def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    # Ensure box1 and box2 are in the shape (N, 4) even if N is 1
    if box1.ndim == 1:
        box1 = box1.reshape(1, 4)
    if box2.ndim == 1:
        box2 = box2.reshape(1, 4)

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (
        (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :]))
        .clip(0)
        .prod(2)
    )

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)

def find_matches(gt_boxes, pred_boxes, iou):
    """
    Given a list of ground truth boxes, predicted boxes and a threshold iou, computes matches and
    returns the number of true positives, false positives and false negatives.
    """
    nb_fp, nb_tp, nb_fn = 0, 0, 0
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)

    # For each prediciton, we check whether we find one or several overlapping ground truth box
    for pred_box in pred_boxes:
        if gt_boxes:
            # Compute matches
            matches = np.array([box_iou(pred_box, gt_box) > iou for gt_box in gt_boxes], dtype=bool)

            # Check if any match exists
            if matches.any():
                nb_tp += 1
                matches = matches.reshape(gt_matches.shape)
                # Logical OR operation in order to update gt_matches with new matches (new True value in the array)
                gt_matches = np.logical_or(gt_matches, matches)  
            else:
                nb_fp += 1
        else:
            nb_fp += 1

    if gt_boxes:
        nb_fn += len(gt_boxes) - np.sum(gt_matches)
    
    return (nb_fp, nb_tp, nb_fn)