import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre, psnr, fsim, sam, uiq, issm
from sewar.full_ref import mse, uqi, ergas, scc, rase, msssim, vifp
import sewar
import numpy as np
from matplotlib import pyplot as plt

max_array = []
min_array = []
ssim_measures = {}

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
        elif steps == "black_white":
            ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
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
            if function_name in ( "mse", "uqi", "ergas", "scc", "rase", "msssim", "vifp"):
                func = getattr(sewar.full_ref, function_name)
                ssim_measures[img_path] = func(test_img, data_img)
            else:
                func = getattr(image_similarity_measures.quality_metrics, function_name)
                ssim_measures[img_path] = func(test_img, data_img)
            # func = getattr(sewar.full_ref, function_name)
    
    ssim_max = calc_closest_val(ssim_measures, True)
    ssim_min = calc_closest_val(ssim_measures, False)

    
    max_array.append(list(ssim_max.values())[0])
    min_array.append(list(ssim_min.values())[0])


def classify_die(test_img_path, data_dir):
    
    similarity = {}
    function_name = "ssim"
    
    test_img = cv2.imread(test_img_path)
    
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        if test_img_path != img_path:
            data_img = cv2.imread(img_path)
            func = getattr(image_similarity_measures.quality_metrics, function_name)
            similarity[img_path] = func(test_img, data_img)
    
    similarity_max = calc_closest_val(similarity, True)

    return os.path.split(list(similarity_max.keys())[0])[1].split(".")[0]

# test_img = cv2.imread('assets/normal_dice/10/854.jpg')
# test_img = cv2.imread('assets/dice 1/Normal/74.jpg')
# test_img = cv2.imread('assets/dice 1/Anomalous/img_17450_cropped.jpg')

# test_img = dilate_image(test_img, 2)

# plt.imshow(test_img)
# plt.show()

# img_17829_cropped.jpg
# anomalous_dice_folder = 'assets/anomalous_dice/2'
# for file in os.listdir(anomalous_dice_folder):
#     classify_die(os.path.join(anomalous_dice_folder, file), "assets/normal_dice")


# data_dir = 'assets/dice 1/Normal'

# die_class = classify_die("assets/anomalous_dice/1/img_17413_cropped.jpg", "assets/normal_dice/classification")

# kernel = 4
# pre_process_steps = ["dilate"]
# normal_dice_folder = 'assets/normal_dice/3'
# metric_name = "psnr"

# rmse, ssim, sre, psnr, fsim, sam, uiq, issm
# mse, uqi, ergas, scc, rase, msssim, vifp

# "sre", "psnr", "sam"

# calculate_similarity('assets/normal_dice/0/16.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)

# i=0
# for folder in os.listdir('assets/normal_dice'):
#     for file in os.listdir(os.path.join('assets/normal_dice', folder)):
#         i+=1
#         calculate_similarity(os.path.join('assets/normal_dice', folder, file), kernel, os.path.join('assets/normal_dice', folder), pre_process_steps, metric_name)

# print(i)
# print(max_array)
# print(min_array)

# print(min(max_array))
# print(min(min_array))

# anomalous_dice_folder = 'assets/anomalous_dice/1'

# max_array = []
# min_array = []
# i=0

# for folder in os.listdir('assets/anomalous_dice'):
#     for file in os.listdir(os.path.join('assets/anomalous_dice', folder)):
#         i+=1
#         anomalous_die_path = os.path.join('assets/anomalous_dice', folder, file)
#         die_class = classify_die(anomalous_die_path, "assets/normal_dice/classification")
#         calculate_similarity(anomalous_die_path, kernel, os.path.join('assets/normal_dice', die_class), pre_process_steps, metric_name)

# print("-------------anomalous dice--------------")

# print(i)
# print(min(max_array))
# print(min(min_array))

# for file in os.listdir(anomalous_dice_folder):
#     calculate_similarity(os.path.join(anomalous_dice_folder, file), kernel, normal_dice_folder, pre_process_steps, metric_name)

# print(max(max_array))
# print(max(min_array))

# calculate_similarity('assets/normal_dice/0/42.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/normal_dice/0/75.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/anomalous_dice/1/img_17450_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/anomalous_dice/1/img_17738_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)

# kernel = 5
# pre_process_steps = ["dilate", "black_white"]
# normal_dice_folder = 'assets/normal_dice/10'
# metric_name = "psnr"

# calculate_similarity('assets/normal_dice/10/854.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/normal_dice/10/800.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/normal_dice/10/845.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/anomalous_dice/img_17583_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)
# calculate_similarity('assets/anomalous_dice/img_18084_cropped.jpg', kernel, normal_dice_folder, pre_process_steps, metric_name)

def compare_anomalous_with_normal(metric_name, pre_process_steps, kernel_size, dice_number):
    
    normal_dice_folder = 'assets/normal_dice/' + str(dice_number)
    anomalous_dice_folder = 'assets/anomalous_dice/' + str((dice_number+1))
    
    for file in os.listdir(normal_dice_folder):
        calculate_similarity(os.path.join(normal_dice_folder, file), kernel_size, normal_dice_folder, pre_process_steps, metric_name)

    print("---------------Normal dice---------------")
    
    print("max_array min : " + str(min(max_array)))
    print("max_array max : " + str(max(max_array)))
    print("min_array min : " + str(min(min_array)))
    print("min_array max : " + str(max(min_array)))
    
    max_array.clear()
    min_array.clear()
    
    for file in os.listdir(anomalous_dice_folder):
        anomalous_die_path = os.path.join(anomalous_dice_folder, file)
        die_class = classify_die(anomalous_die_path, "assets/normal_dice/classification")
        calculate_similarity(anomalous_die_path, kernel_size, os.path.join("assets/normal_dice", die_class), pre_process_steps, metric_name)

    print("---------------Anomalous dice---------------")
    
    print("max_array min : " + str(min(max_array)))
    print("max_array max : " + str(max(max_array)))
    print("min_array min : " + str(min(min_array)))
    print("min_array max : " + str(max(min_array)))

pre_process_steps = ["black_white"]
kernel_size = 4
metric = "ergas"
# rmse, ssim, sre, psnr, fsim, sam, uiq, issm
# mse, uqi, ergas, scc, rase, msssim, vifp
    
compare_anomalous_with_normal(metric,pre_process_steps, kernel_size, 0 )
compare_anomalous_with_normal(metric,pre_process_steps, kernel_size, 5 )