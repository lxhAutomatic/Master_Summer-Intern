# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 00:45:52 2022

@author: Xinhao Lan
"""

import cv2
import numpy as np
import os
from skimage.filters import threshold_otsu
from skimage import measure

# Crop one single Image
def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1
    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]
    return img_crop
# Cropping and output images in one folder
def read_directory(directory_name):
    black_image= ('01_', '06_', '07_', '12_')
    for filename in os.listdir(directory_name):
        filename = filename[13:]
        if filename.startswith(black_image):
            img = cv2.imread(directory_name + "/" + "SIS_00018056_" + filename)
            img = img.astype(np.float32)
            img /= 255
            img_crop = tight_crop(img)   
            img_crop[img_crop>1.]=1. 
            img_crop = img_crop*255.
            cv2.imwrite("D://SteelImage//test1" + "/" + filename, img_crop)
        else:
            img = cv2.imread(directory_name + "/" + "SIS_00018056_" + filename)
            cv2.imwrite("D://SteelImage//test1" + "/" + filename, img)

def image_preprocessing(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cl2 = cl1
    cv2.normalize(cl1, cl2, 0, 128, cv2.NORM_MINMAX, cv2.CV_8U)
    kernel = np.ones((10,10), np.uint8)
    result = cv2.morphologyEx(cl2, cv2.MORPH_BLACKHAT, kernel)
    return result
    
def build_filters():
     filters = []
     lamda = np.pi/2.0
     kern = cv2.getGaborKernel((15, 15), 1.0, 0, lamda, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
     return filters
 
def getGabor(img,filters):
    res = [] 
    for i in range(len(filters)):
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))
    return res  

def img_preprocessing_2(img):
    #灰度归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    # CLAHE有限对比适应性直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    filters = build_filters()
    images = getGabor(img, filters)
    img = images[0]
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U) #灰度归一化
    kernel = np.ones((1,1), np.uint8)
    img = cv2.erode(img, kernel, iterations = 2) 
#read_directory("D://SteelImage//test")
#img = cv2.imread("D://SteelImage//test1//03_0029.jpg", 0)
#image_preprocessing(img)

img_1 = cv2.imread("D://SteelImage//test//SIS_00018056_01_0028.jpg")
img_2 = cv2.imread("D://SteelImage//test//SIS_00018056_02_0002.jpg")
img_3 = cv2.imread("D://SteelImage//test//SIS_00018056_03_0029.jpg")
img_4 = cv2.imread("D://SteelImage//test//SIS_00018056_04_0016.jpg")
img_5 = cv2.imread("D://SteelImage//test//SIS_00018056_05_0020.jpg")
img_6 = cv2.imread("D://SteelImage//test//SIS_00018056_06_0092.jpg")
img_7 = cv2.imread("D://SteelImage//test//SIS_00018056_07_0128.jpg")
img_1 = tight_crop(img_1)
img_2 = tight_crop(img_2)
img_3 = tight_crop(img_3)
img_4 = tight_crop(img_4)
img_5 = tight_crop(img_5)
img_6 = tight_crop(img_6)
img_7 = tight_crop(img_7)
img_preprocessing_2(img_1)
img_preprocessing_2(img_2)
img_preprocessing_2(img_3)
img_preprocessing_2(img_4)
img_preprocessing_2(img_5)
img_preprocessing_2(img_6)
img_preprocessing_2(img_7)
cv2.imshow('1',img_1)
cv2.imshow('2',img_2)
cv2.imshow('3',img_3)
cv2.imshow('4',img_4)
cv2.imshow('5',img_5)
cv2.imshow('6',img_6)
cv2.imshow('7',img_7)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度图
# otsu
#ret, thresh = cv2.threshold(img, 0, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#dst = cv2.adaptiveThreshold(img, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
#ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
#print(ret2)
#cv2.imshow('th2', th2)


