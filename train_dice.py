import numpy as np
from numpy import asarray
from pathlib import Path
import matplotlib.pyplot as plt


def train_dice():

    # this variable will be the outcome of dice number detection function
    face_number = 8

    DATA_PATH = Path(f"data/train/{face_number}")
    print(DATA_PATH)
    results = {}
    statistics = ['avg', 'std']
<<<<<<< HEAD
    margin = [0.98, 1.02]
=======
    if face_number == 3:
        margin = [0.78,1.22]
    elif face_number == 4:
        margin= [0.95,1.05]
    elif face_number == 5:
        margin = [0.98, 1.02]
    elif face_number == 6:
        margin = [0.93, 1.07]
    elif face_number == 7:
        margin = [0.79, 1.21]
    else:
        margin = [0.98,1.02]
>>>>>>> 7e7caa90131d5aa5d0d2dd91802d960e57b65018

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
    X_sigma = 1
    _, axs = plt.subplots(contour_count, 2)
    print(f"{face_number}:{{")
    for stat in statistics:
        print(f" '{stat}':{{")
        for contour in range(contour_count):
            print(f" {contour + 1}:{round(min(results[contour + 1][stat]) * margin[0], 2), round(max(results[contour + 1][stat]) * margin[1], 2)},")

            #all_data = results[contour + 1][stat]
            #with X sigma from average -- this is an option to set a margin
            #print(f" {contour + 1}:{np.mean(all_data) - X_sigma * np.std(all_data), np.mean(all_data) + X_sigma * np.std(all_data)},")

            #absolute value - this is an option to set a margin
            # abs_min_avg = 0
            # abs_max_avg = 0
            # abs_min_std = 10
            # abs_max_std = 5
            # if stat == 'std':
            #     print(f" {contour + 1}:{min(all_data) - abs_min_std, max(all_data) + abs_max_std},")
            # else:
            #     print(f" {contour + 1}:{min(all_data) - abs_min_avg, max(all_data) + abs_max_avg},")

            axs[contour, col].plot(results[contour + 1][stat])
            axs[contour, col].set_title(f'Contour {contour + 1} {stat}')
        print("    },")
        col = + 1
    print("},")

    plt.show(block=True)


# train_dice()
