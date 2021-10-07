import cv2
from image_similarity_measures.quality_metrics import ssim
import os


def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
        closest = min(dict.values())

    for key, value in dict.items():
        if (value == closest):
            result[key] = closest

    return result


def classify_die(img_path: str):

    data_dir = "data/normal_dice/classification"
    similarity = {}

    test_img = cv2.imread(img_path)

    for file in os.listdir(data_dir):
        classification_img_path = os.path.join(data_dir, file)
        if img_path != classification_img_path:
            data_img = cv2.imread(classification_img_path)
            similarity[classification_img_path] = ssim(test_img, data_img)

    similarity_max = calc_closest_val(similarity, True)

    return os.path.split(list(similarity_max.keys())[0])[1].split(".")[0]
