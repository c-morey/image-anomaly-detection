import cv2 as cv
import glob
import numpy as np

im=[]
im_an=[]
im_t=[]

dice_number=10

for imgs in glob.glob(rf"data_isolationforest\normal\{dice_number}\*.jpg"):
    cv_img = cv.imread(imgs)
    i=np.asarray(cv_img)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #img = cv.dilate(i, kernel, iterations=1)
    #ret, img = cv.threshold(img, 140, 255, cv.THRESH_BINARY)
    im.append(i)

    for imgs in glob.glob(rf"data_isolationforest\anomalous_dice\{dice_number}\*.jpg"):
        cv_img_an = cv.imread(imgs)
        ii_an = np.asarray(cv_img_an)
        #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #img_an = cv.dilate(ii_an, kernel, iterations=1)
        #ret, img_an = cv.threshold(img_an, 140, 255, cv.THRESH_BINARY)
        im_an.append(ii_an)

for imgs in glob.glob(rf"data_isolationforest\test\{dice_number}\*.jpg"):
    cv_img_t = cv.imread(imgs)
    ii=np.asarray(cv_img_t)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #img_t = cv.dilate(ii, kernel, iterations=1)
    #ret, img_t = cv.threshold(img_t, 140, 255, cv.THRESH_BINARY)
    im_t.append(ii)


average=[]
std=[]

for i in range(len(im)):
    ave=np.mean(im[i])
    st=np.std(im[i])
    average.append(ave)
    std.append(st)
    r=np.array(average)
    s=np.array(std)
print(r)
print(r.shape)

average_an=[]
std_an=[]

for i in range(len(im_an)):
    ave_an=np.mean(im_an[i])
    st_n = np.std(im_an[i])
    average_an.append(ave_an)
    std_an.append(st_n)
    r_an=np.array(average_an)
    s_an=np.array(std_an)

average_t=[]
std_t=[]

for i in range(len(im_t)):
    ave_t=np.mean(im_t[i])
    st_t = np.std(im_t[i])
    average_t.append(ave_t)
    std_t.append(st_t)
    r_t=np.array(average_t)
    s_t=np.array(std_t)





from sklearn.ensemble import IsolationForest
import argparse
import pickle

import pandas as pd
dataset = pd.DataFrame({'image': im, 'average': list(r),'std':list(s)}, columns=['image', 'average','std'])
dataset_an=pd.DataFrame({'image': im_an, 'average': list(r_an),'std':list(s_an)}, columns=['image', 'average','std'])
dataset_t=pd.DataFrame({'image': im_t, 'average': list(r_t),'std':list(s_t)}, columns=['image', 'average','std'])
random_state = np.random.RandomState(50)

model=IsolationForest(n_estimators=100,max_samples='auto',random_state=random_state)
model.fit(dataset[['average','std']].values)
print(model.get_params())
pred_train=model.predict(dataset[['average','std']].values)
pred_test=model.predict(dataset_t[['average','std']].values)
pred_anomaly=model.predict(dataset_an[['average','std']].values)
print(pred_test)
print(pred_anomaly)

accuracy_validation=100*(list(pred_test).count(1))/len(list(pred_test))
accuracy_anomaly=100*(list(pred_anomaly).count(-1))/len(list(pred_anomaly))
print('accuracy_validation :', accuracy_validation)
print('accuracy_anomaly:', accuracy_anomaly)



