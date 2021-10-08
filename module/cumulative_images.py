import sys
import numpy as np
from numpy import asarray
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def cumulative_images(arguments):

    options = {
        'defect': 'anomalous_dice',
        'std': 'normal_dice',
        'all': ''
    }
    
    if len(arguments)<2:
        all_options = ", ".join(list(options.keys()))
        print("2 argumets are required: the option and die number (between 0 and 10)")
        print(f'List of options available: {all_options}')
        sys.exit()

    option = arguments[0]
    face_number = arguments[1]
    DATA_PATH = Path(f"image-anomaly-detection/data/{options[option]}/{face_number}")
    print(DATA_PATH)
    cumulative_img = np.zeros((128,128))

    for dice_img in DATA_PATH.glob(f"**/*.jpg"):
        print(f"Processing {dice_img.name}")
        image = plt.imread(dice_img)
        data = asarray(image)
        data = (data <= 70).astype(int)
        cumulative_img = cumulative_img + data

    #we convert the cumulative mask to a binary array
    cumulative_img = (cumulative_img >= 1).astype(int)
    plt.imshow(cumulative_img)
    plt.show()

    #the type of numpy array need to be converted to be used by cv2
    cumulative_img = cumulative_img.astype(np.uint8)
    #We crate an empty image same size as cumulative image for masking later
    mask = np.zeros(cumulative_img.shape,np.uint8)
    cumulative_mask = mask.copy()

    #The countour is detected
    contours, hierarchy = cv2.findContours(cumulative_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    init_contour_count = len(contours)
    contour_count = len(contours)
    print(f'Detected contours: {contour_count+1}')

    #Countour representation overall
    with_contours = cv2.drawContours(cumulative_img, contours, -1,(255,0,255),1)
    plt.imshow(with_contours)
    plt.show()

    for cnt in contours:
        print(f"countour_count {contour_count}")
        #One mask is made per contour
        single_mask = mask.copy()
        #Extract only the point within the contour
        cv2.drawContours(single_mask,[cnt],0,1,-1)
        cv2.drawContours(cumulative_mask,[cnt],0,1,-1)
        plt.imshow(single_mask)
        #plt.show()

        #save the mask
        single_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{contour_count}"))
        contour_count -=1

    cumulative_mask = (cumulative_mask < 1).astype(int)
    print(f"countour_count {init_contour_count+1}")
    cumulative_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{init_contour_count+1}"))

