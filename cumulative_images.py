import numpy as np
from numpy import asarray
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def cumulative_images(face_number: int):

    # face number will be the result of dice number similarity function output
<<<<<<< HEAD
=======
    face_number = 4
>>>>>>> 7e7caa90131d5aa5d0d2dd91802d960e57b65018
    DATA_PATH = Path(f"data/train/{face_number}")
    mask_path = Path(f"data/test/{face_number}")
    cumulative_img = np.zeros((128, 128))

    for dice_img in DATA_PATH.glob("**/*.jpg"):
        image = plt.imread(dice_img)
        data = asarray(image)
        data = (data <= 70).astype(int)
        cumulative_img = cumulative_img + data

    # we convert the cumulative mask to a binary array
    cumulative_img = (cumulative_img >= 1).astype(int)

    # the type of numpy array need to be converted to be used by cv2
    cumulative_img = cumulative_img.astype(np.uint8)
    # We crate an empty image same size as cumulative image for masking later
    mask = np.zeros(cumulative_img.shape, np.uint8)
    cumulative_mask = mask.copy()

    # The countour is detected
    contours, hierarchy = cv2.findContours(cumulative_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    init_contour_count = len(contours)
    contour_count = len(contours)

    for cnt in contours:

        # One mask is made per contour
        single_mask = mask.copy()
        # Extract only the point within the contour
        cv2.drawContours(single_mask, [cnt], 0, 1, -1)
        cv2.drawContours(cumulative_mask, [cnt], 0, 1, -1)

        # save the mask
        single_mask.dump(mask_path.joinpath(f"face_number_{face_number}_contour_{contour_count}"))
        single_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{contour_count}"))
        contour_count -= 1

    cumulative_mask = (cumulative_mask < 1).astype(int)

    cumulative_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{init_contour_count + 1}"))
    cumulative_mask.dump(mask_path.joinpath(f"face_number_{face_number}_contour_{init_contour_count+1}"))


def generate_all_cumulative_images():
    for i in range(0, 11):
        cumulative_images(i)
