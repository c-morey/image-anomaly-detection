import cv2
from image_similarity_measures.quality_metrics import ssim
import os
import numpy as np


def calc_closest_val(dict: dict, checkMax: bool) -> dict:
    """We use this function to check for the similarity score

    Args:
        dict ([type]): a dictionary of all similarity score
        checkMax ([type]): bool to specify if we need to return max or min

    Returns:
        min or max from a dictionary depending of the value of checkMax
    """
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
        closest = min(dict.values())

    for key, value in dict.items():
        if (value == closest):
            result[key] = closest

    return result


def classify_die(img) -> str:
    """ We use this function to know which type of die we deal with and what
    is the reference folder for the algorithm

    Args:
        img_path (str)

    Returns:
        str: the type of die (a number between 0 and 10)
    """
    data_dir = "data/normal_dice/classification"
    similarity = {}

    for file in os.listdir(data_dir):
        classification_img_path = os.path.join(data_dir, file)
        data_img = cv2.imread(classification_img_path)
        data_img = np.mean(data_img, axis=2)
        similarity[classification_img_path] = ssim(img, data_img)

    similarity_max = calc_closest_val(similarity, True)

    return os.path.split(list(similarity_max.keys())[0])[1].split(".")[0]
