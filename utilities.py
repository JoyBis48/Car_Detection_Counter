import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Function to create a color mask to detect a specific color range
def create_color_mask(hsv, ranges):
    mask = None
    for i in range(0, len(ranges), 2):
        lower_bound = np.array(ranges[i])
        upper_bound = np.array(ranges[i+1])
        if mask is None:
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            mask += cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

# Function to get the dominant color of a car using HSV color space
def get_car_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "Red": [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        "Orange": [(10, 50, 50), (25, 255, 255)],
        "Yellow": [(25, 50, 50), (35, 255, 255)],
        "Green": [(35, 50, 50), (85, 255, 255)],
        "Blue": [(85, 50, 50), (125, 255, 255)],
        "Purple": [(125, 50, 50), (145, 255, 255)],
        "Pink": [(145, 50, 50), (170, 255, 255)],
        "White": [(0, 0, 200), (180, 30, 255)],
        "Black": [(0, 0, 0), (180, 255, 30)]
    }
    
    max_pixels = 0
    dominant_color = ""
    
    # FindING the color with the most pixels in the ROI
    for color, ranges in color_ranges.items():
        mask = create_color_mask(hsv, ranges)
        num_pixels = cv2.countNonZero(mask)
        if num_pixels > max_pixels:
            max_pixels = num_pixels
            dominant_color = color

    # colour swap
    if dominant_color == "Red":
        return "Blue"
    elif dominant_color == "Blue":
        return "Red"
    
    return dominant_color
    
def get_person_gender(roi, model):
    roi_image = Image.fromarray(roi.astype('uint8'), 'RGB')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        roi_image.save(tmp, format='JPEG')
        tmp_path = tmp.name

    results = model.predict(tmp_path)

    os.remove(tmp_path)

    if len(results) > 0 and len(results[0].boxes) > 0:
        # Accessing the first detection's class ID
        class_id = results[0].boxes.cls[0].item()  # Converting tensor to Python scalar
        
        # Return "Male" if class_id is 0, "Female" if class_id is 1, else return an empty string
        if class_id == 0:
            return "Male"
        elif class_id == 1:
            return "Female"
        else:
            return ""
    else:
        # Return an empty string indicating no detection
        return ""