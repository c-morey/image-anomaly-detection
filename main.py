import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre, psnr, fsim, sam, uiq, issm

test_img = cv2.imread('assets/anomalous_dice/img_17841_cropped.jpg')

ssim_measures = {}
rmse_measures = {}
sre_measures = {}
psnr_measures= {}
fsim_measures = {}
issm_measures = {}
sam_measures = {}
uiq_measures = {}

scale_percent = 100 # percent of original img size

data_dir = 'assets/test'
<<<<<<< HEAD

=======
   
>>>>>>> 48430f1a59ec9c9efe8bfb4724fb6e156873333d
for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    ssim_measures[img_path] = ssim(test_img, data_img)
    rmse_measures[img_path] = rmse(test_img, data_img)
    sre_measures[img_path] = sre(test_img, data_img)
    
    issm_measures[img_path] = issm(test_img, data_img)
    sam_measures[img_path] = sam(test_img, data_img)
    psnr_measures[img_path] = psnr(test_img, data_img)
    fsim_measures[img_path] = fsim(test_img, data_img)


def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
    	closest = max(dict.values())
    else:
    	closest = min(dict.values())
    		
    for key, value in dict.items():
    	print("The difference between ", key ," and the original image is : \n", value)
    	if (value == closest):
    	    result[key] = closest
    	    
    print("The closest value: ", closest)	    
    print("######################################################################")
    return result
    
ssim = calc_closest_val(ssim_measures, True)
rmse = calc_closest_val(rmse_measures, False)
sre = calc_closest_val(sre_measures, True)

issm_max = calc_closest_val(issm_measures, True)
sam_max = calc_closest_val(sam_measures, True)
psnr_max = calc_closest_val(psnr_measures, True)
fsim_max = calc_closest_val(fsim_measures, True)

issm_min = calc_closest_val(issm_measures, False)
sam_min = calc_closest_val(sam_measures, False)
psnr_min = calc_closest_val(psnr_measures, False)
fsim_min = calc_closest_val(fsim_measures, False)


print("The most similar according to SSIM: " , ssim)
print("The most similar according to RMSE: " , rmse)
print("The most similar according to SRE: " , sre)
print("The most similar according to psnr_max: " , psnr_max)
print("The most similar according to fsim_max: " , fsim_max)

