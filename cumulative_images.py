import os
import numpy as np
from numpy import asarray
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def cumulative_images():

    # face number will be the result of dice number similarity function output
    face_number = 1
<<<<<<< HEAD
    DATA_PATH = Path(f"data/normal_dice/{face_number}")
=======
    DATA_PATH = Path(f"/Users/cerenmorey/image-anomaly-detection/data/train/{face_number}")
    mask_path = Path(f"/Users/cerenmorey/image-anomaly-detection/data/test/{face_number}")
>>>>>>> Divided the functions, created train and test dataset
    print(DATA_PATH)
    cumulative_img = np.zeros((128,128))

    for dice_img in DATA_PATH.glob(f"**/*.jpg"):
        print(f"Processing {dice_img.name}")
        image = plt.imread(dice_img)
        data = asarray(image)
        data = (data <= 70).astype(int)
        cumulative_img = cumulative_img + data

    plt.imshow(cumulative_img)
    plt.show(block = True)

    #we convert the cumulative mask to a binary array
    cumulative_img = (cumulative_img >= 1).astype(int)
    plt.imshow(cumulative_img)
    plt.show(block=True)

    im = Image.fromarray((cumulative_img * 255).astype(np.uint8))
    im.save(f"data/normal_dice/Contour/{face_number}_cumulative.jpg")
    
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
<<<<<<< HEAD
    plt.show()
    
    im = Image.fromarray((with_contours * 255).astype(np.uint8))
    im.save(f"data/normal_dice/Contour/{face_number}_cumulative_with_contours.jpg")
=======
    plt.show(block=True)
>>>>>>> Divided the functions, created train and test dataset

    for cnt in contours:
        print(f"countour_count {contour_count}")
        #One mask is made per contour
        single_mask = mask.copy()
        #Extract only the point within the contour
        cv2.drawContours(single_mask,[cnt],0,1,-1)
        cv2.drawContours(cumulative_mask,[cnt],0,1,-1)



        #save the mask
        single_mask.dump(mask_path.joinpath(f"face_number_{face_number}_contour_{contour_count}"))
        single_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{contour_count}"))
        contour_count -=1

    cumulative_mask = (cumulative_mask < 1).astype(int)
    print(f"countour_count {init_contour_count+1}")
    cumulative_mask.dump(DATA_PATH.joinpath(f"face_number_{face_number}_contour_{init_contour_count + 1}"))
    cumulative_mask.dump(mask_path.joinpath(f"face_number_{face_number}_contour_{init_contour_count+1}"))

cumulative_images()


cumulative_images()