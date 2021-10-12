import numpy as np
from numpy import asarray
from pathlib import Path
import matplotlib.pyplot as plt
from classify import classify_die
from config import thresholds
from cv2 import imdecode


<<<<<<< HEAD
def find_defect(face_number):
=======
def find_defect():

    #this variable willl be the outcome of dice number detection function
    face_number = 8
>>>>>>> 7e7caa90131d5aa5d0d2dd91802d960e57b65018

    DATA_PATH = Path(f"data/test/{face_number}")
    counter = {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
    results = {}
    defect_list = {'avg': [], 'std': []}
    statistics = ['avg', 'std']

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

            if contour_number not in results:
                results[contour_number] = {'std': [], 'avg': []}

            results[contour_number]['avg'].append(mx.mean())
            results[contour_number]['std'].append(mx.std())

            for stat in statistics:
                LL = thresholds[face_number][stat][contour_number][0]
                UL = thresholds[face_number][stat][contour_number][1]

                if stat == 'avg':
                    value = mx.mean()
                else:
                    value = mx.std()

                if value < LL or value > UL:
                    if dice_img.name not in defect_list[stat]:
                        defect_list[stat].append(dice_img.name)

        # this loop will save the true positive-negatives for evaluation matrix
        detected = False
        if "cropped" in dice_img.name:
            for stat in statistics:
                if dice_img.name in defect_list[stat]:
                    counter["TP"] = counter["TP"] + 1
                    detected = True
                    break
            if not detected:
                counter["FN"] = counter["FN"] + 1
        else:
            for stat in statistics:
                if dice_img.name in defect_list[stat]:
                    counter["FP"] = counter["FP"] + 1
                    detected = True
                    break
            if not detected:
                counter["TN"] = counter["TN"] + 1

    # Print the results and plot--------------
    contour_count = len(results.keys())
    col = 0
    _, axs = plt.subplots(contour_count, 2)
    for stat in statistics:
        for contour in range(contour_count):
            axs[contour, col].plot(results[contour + 1][stat])
            LL = thresholds[face_number][stat][contour+1][0]
            UL = thresholds[face_number][stat][contour+1][1]
            axs[contour, col].hlines(LL, 0, len(results[contour+1][stat]))
            axs[contour, col].hlines(UL, 0, len(results[contour+1][stat]))
            axs[contour, col].set_title(f'Contour {contour+1} {stat}')
        col =+ 1

    plt.show(block=True)

    # Show the detected defective images----------------
    for stat in statistics:
        if len(defect_list[stat]) != 0:
            print(f"{len(defect_list[stat])} defects found in {stat} list")
            for defect in defect_list[stat]:
                print(defect)
                def_img = plt.imread(f"data/test/{face_number}/{defect}")
                plt.imshow(def_img)
                plt.show(block=True)
        else:
            print(f"No defects found with {stat}")

    # pickling the counter dict to use in evaluation.py
    with open(f"counter_{face_number}", "wb") as handle:
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)


def is_defect(image_bytes):

    image = imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    data = asarray(image)

    # this variable willl be the outcome of dice number detection function
    face_number = int(classify_die(data))
    # face_number = 1
    print("---------------- face number ----------------------")
    print(face_number)

    DATA_PATH = Path(f"data/train/{face_number}")
    results = {}
    statistics = ['avg', 'std']

    for contour in DATA_PATH.glob("**/face_number_*"):
        print(contour)
        contour_number = int(contour.name.split("_")[-1])
        mask = np.load(contour, allow_pickle=True)
        mask = (mask == 0).astype(int)
        mx = np.ma.masked_array(data, mask=mask)

        if contour_number not in results:
            results[contour_number] = {'std': [], 'avg': []}

        results[contour_number]['avg'].append(mx.mean())
        results[contour_number]['std'].append(mx.std())

        for stat in statistics:
            LL = thresholds[face_number][stat][contour_number][0]
            UL = thresholds[face_number][stat][contour_number][1]

            if stat == 'avg':
                value = mx.mean()
            else:
                value = mx.std()

            if value < LL or value > UL:
<<<<<<< HEAD
                return True, face_number
=======
                return True

    return False

    # Print the results and plot--------------
    contour_count = len(results.keys())
    col = 0
    _, axs = plt.subplots(contour_count, 2)
    for stat in statistics:
        for contour in range(contour_count):
            axs[contour, col].plot(results[contour + 1][stat])
            LL = thresholds[face_number][stat][contour+1][0]
            UL = thresholds[face_number][stat][contour+1][1]
            axs[contour, col].hlines(LL, 0, len(results[contour+1][stat]))
            axs[contour, col].hlines(UL, 0, len(results[contour+1][stat]))
            axs[contour, col].set_title(f'Contour {contour+1} {stat}')
        col =+ 1

    plt.show(block=True)

    # Show the detected defective images----------------
    for stat in statistics:
        if len(defect_list[stat]) != 0:
            print(f"{len(defect_list[stat])} defects found in {stat} list")
            for defect in defect_list[stat]:
                print(defect)
                def_img = plt.imread(f"data/test/{face_number}/{defect}")
                def_img = plt.imread(img_path)
                plt.imshow(def_img)
                plt.show(block=True)
        else:
            print(f"No defects found with {stat}")


>>>>>>> 7e7caa90131d5aa5d0d2dd91802d960e57b65018

    return False, face_number
