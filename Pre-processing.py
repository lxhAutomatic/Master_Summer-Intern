# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 00:45:52 2022

@author: Xinhao Lan
"""
# built-in modules
import os

# third-party modules
import cv2
import numpy as np


def build_filters():
    """
    Function to build the gabor filter.

    :return filters: list. The type of the each item in the list is numpy.ndarray.
    """
    filters = []
    lamda = np.pi/2.0
    kern = cv2.getGaborKernel((15, 15), 1.0, 0, lamda, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)
    return filters
 
def getGabor(img,filters):
    """
    Function for the filtering with the use of Gabor filter.

    :param img: np.ndarray. Input image after the CLAHE.
    :param filters: list. The type of the each item in the list is numpy.ndarray.

    :return res: list. List of images after the filtering of different filters.
    """
    res = [] 
    for i in range(len(filters)):
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))
    return res  

def preprocessing(img):
    """
    Function for the pre-processing.

    :param img: np.ndarray. Input images for preprocessing.

    :return img: np.ndarray. Output images after preprocessing.
    """
    # Change the image to the grey image to make algorithm in cv2 can be applied
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # Use the Gabor filter to reduce the noise.
    filters = build_filters()
    images = getGabor(img, filters)
    img = images[0]
    # Normalize the image again
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U) 
    # Erosion
    kernel = np.ones((1,1), np.uint8)
    img = cv2.erode(img, kernel, iterations = 2)
    return img

def read_directory(input_directory_name, output_directory_name_1, output_directory_name_2):
    """
    Read the image in the folder and output the defects image in directory 1 and other raw images in directory 2.

    :param input_directory_name: string. Path of the input folder which contains original images.
    :param output_directory_name_1: string. Path of the output folder which will contain defects images.
    :param output_directory_name_2: string. Path of the output folder which will contain images with no defects.
    """
    for filename in os.listdir(input_directory_name):
        img = cv2.imread(input_directory_name + "//" + filename)
        if img.shape[0]>img.shape[1]:
            img = preprocessing(img)
            cv2.imwrite(output_directory_name_1 + "//" + filename, img)
        else:
            cv2.imwrite(output_directory_name_2 + "//" + filename, img)
            
"""
input_dir = ''
output_dir_1 = ''
output_dir_2 = ''
read_direcatory(input_dir, output_dir_1, output_dir_2)
"""