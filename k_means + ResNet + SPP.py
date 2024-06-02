# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:22:20 2022

@author: Xinhao Lan
"""

#import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
import math
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#from spp_layer import spatial_pyramid_pool
import torchvision.models as models
"""
class SPP_NET(nn.Module):
    '''
    A CNN model which adds spp layer so that we can input multi-size tensor
    '''
    def __init__(self, opt, input_nc, ndf=64,  gpu_ids=[]):
        super(SPP_NET, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_num = [4,2,1]
        
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(10752,4096)
        self.fc2 = nn.Linear(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.LReLU1(x)

        x = self.conv2(x)
        x = F.leaky_relu(self.BN1(x))

        x = self.conv3(x)
        x = F.leaky_relu(self.BN2(x))
        
        x = self.conv4(x)
        # x = F.leaky_relu(self.BN3(x))
        # x = self.conv5(x)
        spp = spatial_pyramid_pool(x,1,[int(x.size(2)),int(x.size(3))],self.output_num)
        # print(spp.size())
        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)
        s = nn.Sigmoid()
        output = s(fc2)
        return output
"""

input_dir = 'D://SteelImage//test1'
glob_dir = input_dir + '/*.jpg'

images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
#need to use SSP tp do the resize
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images), -1) / 255)

model = models.resnet34(pretrained=True)
#model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
predictions = model(torch.from_numpy(images.reshape(-1, 3, 224, 224)))
pred_images = predictions.reshape(images.shape[0], -1)
pred_images = pred_images.detach().numpy()

def find_best_k(pred_images):
    sil = []
    kl = []
    kmax = 3
    for k in range(2, kmax + 1):
        kMeans = KMeans(n_clusters=k).fit(pred_images)
        labels = kMeans.labels_ 
        sil.append(silhouette_score(pred_images, labels, metric='euclidean')) 
        kl.append(k)
    bestK = kl[sil.index(max(sil))]
    return bestK

k = find_best_k(pred_images)
#k = 4
kMeansModel = KMeans(n_clusters=k)
kMeansModel.fit(pred_images)
label_pred = kMeansModel.labels_  
kPredictions = kMeansModel.predict(pred_images)
print(kPredictions)

for i in range(1,k+1):
    name="D://SteelImage//test1//class" + str(i)
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir("D://SteelImage//test1//class" + str(i))
for i in range(len(paths)):
    for j in range(0,k):
        if kPredictions[i] == j:
            shutil.copy(paths[i], "D://SteelImage//test1//class"+str(j+1))
