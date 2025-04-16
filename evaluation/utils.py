import datetime
import re
import os

def is_image(image_path):
    return os.path.isfile(image_path) and os.path.splitext(image_path)[-1].lower() in EXT


def parse_date_from_filename(filename):
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
        file_datetime = datetime.datetime(year, month, day, hour, minute, second)
        return file_datetime
    
    return None