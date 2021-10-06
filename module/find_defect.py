import sys
import numpy as np
from numpy import asarray
from pathlib import Path
import matplotlib.pyplot as plt
from config import thresholds

def find_defect(arguments):
    
    #Check the arguments--------------------
    options = ['train','detect']
    if len(arguments)<2:
        all_options = ", ".join(options)
        print("2 arguments are required")
        print(f"An option is needed, possible values: {all_options}")
        print("The face number between 0 and 10 need to be provided as second argument")
        print("USAGE: python3 -m dice_analysis FD detect 10")
        sys.exit()

    if arguments[0] not in options:
        all_options = ", ".join(options)
        print(f"An option is needed, possible values: {all_options}")
        print("USAGE: python3 -m dice_analysis FD detect 10")
        sys.exit()

    #Define the parameters-------------------------
    option = arguments[0]
    face_number = int(arguments[1])

    DATA_PATH = Path(f"image-anomaly-detection/data/normal_dice/{face_number}")
    print(DATA_PATH)
    results = {}
    defect_list = {'avg':[],'std':[]}
    statistics = ['avg','std']
    margin= [0.98,1.02]

    #loop over images ------------
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
                results[contour_number] = {'std':[],'avg':[]}
            
            results[contour_number]['avg'].append(mx.mean())
            results[contour_number]['std'].append(mx.std())

            if option != "train":
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

    #Print the results and plot--------------
    contour_count = len(results.keys())
    col = 0
    _, axs = plt.subplots(contour_count, 2)
    print(f"{face_number}:{{")
    for stat in statistics:
        print(f"    '{stat}':{{")
        for contour in range(contour_count):
            print(f" {contour+1}:{round(min(results[contour+1][stat])*margin[0],2),round(max(results[contour+1][stat])*margin[1],2)},")
            axs[contour,col].plot(results[contour+1][stat])
            if option != "train":
                LL = thresholds[face_number][stat][contour+1][0]*margin[0]
                UL = thresholds[face_number][stat][contour+1][1]*margin[1]
                axs[contour,col].hlines(LL,0,len(results[contour+1][stat]))
                axs[contour,col].hlines(UL,0,len(results[contour+1][stat]))
            axs[contour,col].set_title(f'Contour {contour+1} {stat}')
        print("    },")
        col =+ 1
    print("},")
    
    plt.show()

    #Show the detected defective images----------------
    if option != "train":
        for stat in statistics:
            if len(defect_list[stat]) != 0:
                print(f"{len(defect_list[stat])} defects found in {stat} list")
                for defect in defect_list[stat]:
                    print(defect)
            else:
                print(f"No defects found with {stat}")
