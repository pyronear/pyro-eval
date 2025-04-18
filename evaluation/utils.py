import datetime
import random
import re
import os

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
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6))
        
        # Create datetime object
        file_datetime = datetime(year, month, day, hour, minute, second)
        return file_datetime
    
    return None

def replace_bool_values(data):
    '''
    Replace True/False by "true"/"false" to be able to dump a dictionnary in a json file.
    '''
    if isinstance(data, dict):
        return {key: replace_bool_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_bool_values(item) for item in data]
    elif isinstance(data, np.bool_):
        return "True" if data else "False"
    else:
        return data

def compute_f1_score(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rand_suffix = random.randint(1000, 9999)
    return f"run-{timestamp}-{rand_suffix}"