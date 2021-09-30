import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('assets/dice 1/Anomalous/img_17450_cropped.jpg')

plt.imshow(img)
plt.show()

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

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)


plt.imshow(thresh1)
plt.show()

img = cv.imread('assets/anomalous_dice/img_17829_cropped.jpg')
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)


plt.imshow(thresh1)
plt.show()

print(thresh1)
# print(img)