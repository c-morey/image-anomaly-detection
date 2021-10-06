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

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

kernel_size = 2


img = cv2.imread('assets/anomalous_dice/6/img_17829_cropped.jpg')
# img = cv2.imread('assets/anomalous_dice/6/img_17969_cropped.jpg')
# img = cv2.imread('assets/anomalous_dice/0/img_17450_cropped.jpg')
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

plt.imshow(img)
plt.show()

kernel = np.ones((kernel_size,kernel_size),np.uint8)
# img = cv2.dilate(img,kernel,iterations = 1)

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# img = cv2.filter2D(img, -1, kernel)
# img = cv2.dilate(img,kernel,iterations = 1)

# img = unsharp_mask(img)

# plt.imshow(img)
# plt.show()

# kernel = np.ones((kernel_size,kernel_size),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)

plt.imshow(img)
plt.show()

ret,img = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
# img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# img = erosion = cv2.erode(img,kernel,iterations = 1)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

plt.imshow(img)
plt.show()
# plt.imshow(th2)
# plt.show()
# plt.imshow(th3)
# plt.show()

# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)




img = cv2.imread('assets/5/Normal dice/655.jpg')
kernel = np.ones((kernel_size,kernel_size),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


# plt.imshow(thresh1)
# plt.show()

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