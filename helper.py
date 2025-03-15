import re
import cv2
import numpy as np


def read_calibration(file_path):
    """ Read camera intrinsic matrix from the calibration file. """
    with open(file_path, 'r') as file:
        content = file.read().replace('\n', '').replace(';', '')
        K_values = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if len(K_values) < 9:
            raise ValueError("Calibration file does not contain enough values for a 3x3 intrinsic matrix.")
        K = np.array(list(map(float, K_values[:9]))).reshape(3, 3)
    return K

def load_images(image_paths):
    """ Load all images from specified paths as color images. """
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    for idx, img in enumerate(images):
        if img is None:
            raise FileNotFoundError(f"Image at path {image_paths[idx]} could not be loaded.")
    return images

def extract_colors_from_image(image, points):
    """Extract color values (BGR) from the image based on 2D points."""
    h, w = image.shape[:2]
    colors = []

    for point in points:
        x, y = int(point[0]), int(point[1])
        # Ensure the point is within image bounds
        if 0 <= x < w and 0 <= y < h:
            color = image[y, x]  # OpenCV uses (y, x) indexing
            colors.append(color / 255.0)  # Normalize to [0, 1] for Open3D
        else:
            colors.append([0, 0, 0])  # Black if point is out of bounds

    return np.array(colors)