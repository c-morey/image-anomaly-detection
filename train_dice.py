import numpy as np
from numpy import asarray
from pathlib import Path
import matplotlib.pyplot as plt
from config import thresholds
import os


def train_dice():

    # this variable will be the outcome of dice number detection function
    face_number = 1

    DATA_PATH = Path(f"/Users/cerenmorey/image-anomaly-detection/data/train/{face_number}")
    print(DATA_PATH)
    results = {}
    statistics = ['avg', 'std']
    margin= [0.98,1.02]

    # loop over images ------------
    for dice_img in DATA_PATH.glob(f"**/*.jpg"):
        print(f"Processing {dice_img.name} from face number {face_number}")
        image = plt.imread(dice_img)
        data = asarray(image)

        for contour in dice_img.parent.glob("**/face_number_*"):
            print(contour)
            contour_number = int(contour.name.split("_")[-1])
            mask = np.load(contour, allow_pickle=True)
            mask = (mask == 0).astype(int)
            mx = np.ma.masked_array(data, mask=mask)

            # plt.imshow(mx)
            # plt.show()

            if contour_number not in results:
                results[contour_number] = {'std': [], 'avg': []}

            results[contour_number]['avg'].append(mx.mean())
            results[contour_number]['std'].append(mx.std())


    # Print the results and plot--------------
    contour_count = len(results.keys())
    col = 0
    _, axs = plt.subplots(contour_count, 2)
    print(f"{face_number}:{{")
    for stat in statistics:
        print(f" '{stat}':{{")
        for contour in range(contour_count):
            print(f" {contour + 1}:{round(min(results[contour + 1][stat]) * margin[0], 2), round(max(results[contour + 1][stat]) * margin[1], 2)},")
            axs[contour, col].plot(results[contour + 1][stat])
            axs[contour, col].set_title(f'Contour {contour + 1} {stat}')
        print("    },")
        col = + 1
    print("},")

    plt.show(block=True)


train_dice()