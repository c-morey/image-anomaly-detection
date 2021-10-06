import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
# img = cv2.imread('assets/anomalous_dice/img_17450_cropped.jpg')

# plt.imshow(img)
# plt.show()

# kernel = np.ones((5,5),np.uint8)
# dilation = cv.dilate(img,kernel,iterations = 1)
# erosion = cv.erode(img,kernel,iterations = 1)

# plt.imshow(dilation)
# plt.show()

# img = cv.imread('assets/dice 1/Normal/39.jpg')

# plt.imshow(img)
# plt.show()

# kernel = np.ones((3,3),np.uint8)
# dilation = cv.dilate(img,kernel,iterations = 1)

# plt.imshow(dilation)
# plt.show()

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


# plt.imshow(thresh1)
# plt.show()
kernel_size = 4


img = cv2.imread('assets/5/Anomalous dice/img_17829_cropped.jpg')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

plt.imshow(thresh1)
plt.show()

img = cv2.imread('assets/5/Anomalous dice/img_17829_cropped.jpg')
# kernel = np.ones((kernel_size,kernel_size),np.uint8)
# img = cv2.dilate(img,kernel,iterations = 1)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im = cv2.filter2D(img, -1, kernel)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


plt.imshow(thresh1)
plt.show()

img = cv2.imread('assets/5/Normal dice/655.jpg')
kernel = np.ones((kernel_size,kernel_size),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


plt.imshow(thresh1)
plt.show()

# print(img)

# normal_dice_folder = 'assets/normal_dice/0'
# img_array = []
# i=0
# for file in os.listdir(normal_dice_folder):
#     if i < 5:
#         i += 1
#         img_path = os.path.join(normal_dice_folder, file)
#         img = cv2.imread(img_path)
#         img_array.append(img_array)

# # ims = np.array(img_array)
# imave = np.average(img_array,axis=0)

# plt.imshow(imave)
# plt.show()