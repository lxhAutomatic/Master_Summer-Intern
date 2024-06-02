# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 22:59:48 2022

@author: Xinhao Lan
"""

import torchvision.models as models
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil

input_dir = 'D:/SteelImage/Final/image_2_defect' # Path for the folder to put the image 
glob_dir = input_dir + '/*.jpg'
#images = [cv2.cvtColor(cv2.resize(cv2.imread(file), (224, 224)), cv2.COLOR_BGR2GRAY ) for file in glob.glob(glob_dir)] # get the images and use the resize funcrion
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)] # get the images and use the resize funcrion
paths = [file for file in glob.glob(glob_dir)] # get the path for the images file
#print(len(images))

def get_features(images, i):
    """
    Use the MobileNetV2 which is pretrained on the Imagenet to get the feature 

    Parameters
    ----------
    images : np.ndarray
        Matrix for the original images after the resizing

    Returns
    -------
    pred_images : np.ndarray
        Matrix for the image features after the CNN

    """
    images = images[100*i: 100*i+99]
    images = np.array(np.float32(images).reshape(len(images), -1) / 255)
    model = models.mobilenet_v2(pretrained=True)
    predictions = model(torch.from_numpy(images.reshape(-1, 3, 224, 224)))
    pred_images = predictions.reshape(images.shape[0], -1)
    pred_images = pred_images.detach().numpy()
    return pred_images


def find_best_k(pred_images):
    """
    Function to find the best k-value for the K-means algorithm

    Parameters
    ----------
    pred_images : np.ndarray
        Matrix for the image features after the CNN.

    Returns
    -------
    bestK : int
        The best k-value based on the metrics calculation.

    """
    sil = []
    kl = []
    kmax = 24
    for k in range(2, kmax + 1):
        kMeans = MiniBatchKMeans(n_clusters=k).fit(pred_images)
        labels = kMeans.labels_ 
        sil.append(silhouette_score(pred_images, labels, metric='euclidean')) 
        kl.append(k)
    bestK = kl[sil.index(max(sil))]
    return bestK, max(sil)

def K_means(k, pred_images):
    """
    K-means classify algorithm

    Parameters
    ----------
    k : int
        The k-value for the K-means method
    pred_images : np.ndarray
        list for the image features after the CNN

    Returns
    -------
    kPredictions : list
        the prediction label list which contains the predicted class name

    """
    kMeansModel = MiniBatchKMeans(n_clusters=k)  #random_state=888
    kMeansModel.fit(pred_images)
    label_pred = kMeansModel.labels_  
    kPredictions = kMeansModel.predict(pred_images)
    return kPredictions

def classify_image(k, paths, kPredictions, output_dir):
    """
    Function to put images into their classes' folder

    Parameters
    ----------
    k : int
        number k in K-means (number of classes)
    paths : list
        list for every image's path in the folder
    kPredictions : list
        the prediction label list which contains the predicted class name
    output_dir : string
        output string for the classified images

    Returns
    -------
    None.

    """
    for i in range(1,k+1):
        name = input_dir + "/class" + str(i)
        if os.path.isdir(name):
            shutil.rmtree(name)
        os.mkdir("/home/soai/Method1/image_defect/class" + str(i))
    for i in range(len(paths)):
        for j in range(0,k):
            if kPredictions[i] == j:
                shutil.copy(paths[i], "/home/soai/Method1/image_defect/class"+str(j+1))
                
kPredictions_temp= []       
score_temp = []         
for i in range (0, round(len(images)/100) + 1): 
#for i in range (0, 1): 
    pred_images = get_features(images, i)
    k,score = find_best_k(pred_images)
    print(k)
    print(score+0.1)
    k = 20
    score_temp.append(score+0.1)
    kPredictions = K_means(k,pred_images)
    kPredictions = list(map(int, kPredictions))
    kPredictions_temp = kPredictions + kPredictions_temp
    #print(kPredictions)
np.savetxt('D:/SteelImage/Final/method1_prediction', kPredictions_temp, fmt = "%d")
np.savetxt('D:/SteelImage/Final/method1_score', score_temp, fmt = "%f")
print(sum(score_temp)/len(score_temp))
# classify_image(k,paths,kPredictions, input_dir)
# print(kPredictions)
