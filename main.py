import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre, psnr, fsim, sam, uiq, issm
from sewar.full_ref import mse, uqi, ergas, scc, rase, msssim, vifp
import numpy as np
from matplotlib import pyplot as plt

def dilate_image(img, kernel_size):

    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    dilate_image = cv2.dilate(img,kernel,iterations = 1)
    
    return dilate_image


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


def execute_pre_process(img, pre_process_steps, kernel):
    for steps in pre_process_steps:
        if steps == "dilate":
            img = cv2.dilate(img,kernel,iterations = 1)
    
    return img

def calculate_similarity(test_img_path, kernel_size, data_dir, pre_process_steps, function_name):
    
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    
    test_img = cv2.imread(test_img_path)
    test_img = execute_pre_process(test_img, pre_process_steps, kernel)
    
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        if test_img_path != img_path:
            data_img = cv2.imread(img_path)
            data_img = execute_pre_process(data_img, pre_process_steps, kernel)
            # data_img = dilate_image(data_img, 2)
            func = getattr(image_similarity_measures.quality_metrics, function_name)
            # func = getattr(sewar.full_ref, function_name)
            ssim_measures[img_path] = func(test_img, data_img)
            # ssim_measures[img_path] = mse(test_img, data_img)
    
    ssim_max = calc_closest_val(ssim_measures, True)
    ssim_min = calc_closest_val(ssim_measures, False)

    print("The max: " , ssim_max)
    print("The min: " , ssim_min)


# test_img = cv2.imread('assets/normal_dice/10/854.jpg')
# test_img = cv2.imread('assets/dice 1/Normal/74.jpg')
# test_img = cv2.imread('assets/dice 1/Anomalous/img_17450_cropped.jpg')

# test_img = dilate_image(test_img, 2)

# plt.imshow(test_img)
# plt.show()

# img_17829_cropped.jpg
ssim_measures = {}

# data_dir = 'assets/dice 1/Normal'

kernel = 5
pre_process_steps = []
normal_dice_folder = 'assets/normal_dice/10'
metric_name = "psnr"
# rmse, ssim, sre, psnr, fsim, sam, uiq, issm
# mse, uqi, ergas, scc, rase, msssim, vifp

# "sre", "psnr", "sam"

calculate_similarity('assets/normal_dice/10/854.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
calculate_similarity('assets/normal_dice/10/800.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
calculate_similarity('assets/normal_dice/10/845.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
calculate_similarity('assets/dice 6/img_17583_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
calculate_similarity('assets/dice 6/img_18084_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)