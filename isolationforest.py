import cv2 as cv
import glob
import numpy as np

from sklearn.ensemble import IsolationForest
import argparse
import pickle

import pandas as pd
dice_number=10

def load_image(dice_number : int):
    ''' this function loads normal and anomalouse image as train, test, and anomaly image
     and return array for each images'''

    normal_image=[]
    anomaly_image=[]
    validation_image=[]

    for imgs in glob.glob(rf"data_isolationforest\normal\{dice_number}\*.jpg"):
        cv_img = cv.imread(imgs)
        i=np.asarray(cv_img)
        #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #img = cv.dilate(i, kernel, iterations=1)
        #ret, img = cv.threshold(img, 140, 255, cv.THRESH_BINARY)
        normal_image.append(i)

    for imgs in glob.glob(rf"data_isolationforest\anomalous_dice\{dice_number}\*.jpg"):
        cv_img_an = cv.imread(imgs)
        ii_an = np.asarray(cv_img_an)
        #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #img_an = cv.dilate(ii_an, kernel, iterations=1)
        #ret, img_an = cv.threshold(img_an, 140, 255, cv.THRESH_BINARY)
        anomaly_image.append(ii_an)

    for imgs in glob.glob(rf"data_isolationforest\test\{dice_number}\*.jpg"):
        cv_img_t = cv.imread(imgs)
        ii=np.asarray(cv_img_t)
        #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #img_t = cv.dilate(ii, kernel, iterations=1)
        #ret, img_t = cv.threshold(img_t, 140, 255, cv.THRESH_BINARY)
        validation_image.append(ii)

    return (normal_image, anomaly_image, validation_image)

def mean_std(normal_image, anomaly_image, validation_image):

    normal_average=[]
    normal_std=[]

    for i in range(len(normal_image)):
        ave=np.mean(normal_image[i])
        st=np.std(normal_image[i])
        normal_average.append(ave)
        normal_std.append(st)
        normal_ave=np.array(normal_average)
        normal_s=np.array(normal_std)



    anomaly_im_average=[]
    anomaly_im_std=[]

    for i in range(len(anomaly_image)):
        ave_an=np.mean(anomaly_image[i])
        st_n = np.std(anomaly_image[i])
        anomaly_im_average.append(ave_an)
        anomaly_im_std.append(st_n)
        anomaly_im_av=np.array(anomaly_im_average)
        anomaly_im_st=np.array(anomaly_im_std)

    average_validation=[]
    std_validation=[]

    for i in range(len(validation_image)):
        ave_t=np.mean(validation_image[i])
        st_t = np.std(validation_image[i])
        average_validation.append(ave_t)
        std_validation.append(st_t)
        average_valid=np.array(average_validation)
        std_valida=np.array(std_validation)

    return(normal_ave,anomaly_im_av,average_valid,normal_s, anomaly_im_st ,std_valida)


def create_dataset(dice_number:int):
    normal_image, anomaly_image, validation_image=load_image(dice_number)

    normal_average, anomaly_im_average, average_validation, normal_std, anomaly_im_std, std_validation=mean_std(normal_image, anomaly_image, validation_image)

    dataset = pd.DataFrame({'image':normal_image , 'average': list(normal_average),'std':list(normal_std)}, columns=['image', 'average','std'])
    dataset_an=pd.DataFrame({'image': anomaly_image, 'average': list(anomaly_im_average),'std':list(anomaly_im_std)}, columns=['image', 'average','std'])
    dataset_t=pd.DataFrame({'image': validation_image, 'average': list(average_validation),'std':list(std_validation)}, columns=['image', 'average','std'])

    return dataset, dataset_an, dataset_t

def isolation_forest(dice_number:int):
    dataset, dataset_an, dataset_t=create_dataset(dice_number)


    random_state = np.random.RandomState(50)
    model=IsolationForest(n_estimators=100,max_samples='auto',random_state=random_state)
    model.fit(dataset[['average','std']].values)
    #print(model.get_params())
    pred_train=model.predict(dataset[['average','std']].values)
    pred_test=model.predict(dataset_t[['average','std']].values)
    pred_anomaly=model.predict(dataset_an[['average','std']].values)
    #print(pred_test)
    #print(pred_anomaly)

    accuracy_validation=100*(list(pred_test).count(1))/len(list(pred_test))
    accuracy_anomaly=100*(list(pred_anomaly).count(-1))/len(list(pred_anomaly))
    #print('accuracy_validation :', accuracy_validation)
    #print('accuracy_anomaly:', accuracy_anomaly)

    return f'''normal_dice_detection:{pred_test},
             anomaly_detection: {pred_anomaly},
             accuracy_validation:{accuracy_validation},
              accuracy_anomaly: {accuracy_anomaly}'''



print(isolation_forest(dice_number))

