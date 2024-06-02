# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 14:12:47 2022

@author: Xinhao Lan
"""
# built-in modules
import glob
import random
import os

# third-party modules
import cv2
import numpy as np

def find_front_image(img_dir):
    """
    This function is used to substract the background and extract the front image.

    :param img_dir: string. The input direcatory of the image/folder.

    :return mask: numpy.ndarray. Matrix of the front image (exclude the background).
    """
    # if use this function to deal with many images, use the code below instead. Remember to change the index.
    #for filename in os.listdir(img_dir):
    #    img = cv2.imread(img_dir + "//" + filename)
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    img2 = cv2.Canny(img,50,150)
    cv2.imshow('canny', img2)
    rows,cols = img2.shape
    SIZE = 3 
    P = int(SIZE/2)
    BLACK = 0
    WHITE = 255
    BP = []
 
    for row in range(P, rows-P, 1):
        for col in range(P, cols-P, 1):
            if (img2[row, col] == WHITE).all():
                kernal = []
                for i in range(row-P, row+P+1, 1):
                    for j in range(col-P, col+P+1, 1):
                        kernal.append(img2[i, j])
                        if (img2[i, j] == BLACK).all():
                            BP.append([i, j])
 
    uniqueBP = np.array(list(set([tuple(c) for c in BP])))
 
    for x,y in uniqueBP:
        img2[x,y] = 255
 
    mask = cv2.bitwise_and(img,img,mask=img2)
    cv2.imshow('front image', mask)
    return mask


def replace_image(mask, mean_image, mean_image_path):
    """
    This function is used to fill the mean image in the blank area.
    
    :param mask: numpy.ndarray. Matrix of the front image (exclude the background).
    :param mean_image: numpy.ndarray. Mean image of all images in the folder (resize to 224*224).
        
    return mask: numpy.ndarray. Matrix of the image after filling the mean image.
    """
    cv2.imwrite(mean_image_path, mean_image)
    rows, cols = mask.shape
    for i in range(0, rows):
        for j in range (0, cols):
            if(mask[i, j] == 0):
                m = random.randint(0, 223)
                n = random.randint(0, 223)
                pixel = addNoise(mean_image)[m, n]
                mask[i, j] = pixel
    return mask

def getMeanImage(dir, regex_list):
    """
    Calculates the mean image from the given list of images.
    
    :param dir: directory to read images from.
    :param regex_list: give a list of regex to read desired files, done to read only selected files.
    
    :return image: Returns the numpy array of averaged image.
    """
    images_list = []
    for reg in regex_list:
        for img_path in glob.glob(dir + "/" + reg):
            # In my code I use the pretrained model, so the size of the input image is (224,224) 
            img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (224, 224))
            images_list.append(img)
    mean_image = np.mean(images_list, axis=0)
    cv2.normalize(mean_image, mean_image, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    return mean_image

def addNoise(img):
    """
    Add salt noise and pepper noise on the image to make sure that those parts will not be the feature.

    :param img: numpy.ndarray. The matrix for the image before adding noise.

    :return img: numpy.ndarray. The matrix for the image after adding noise.
    """
    row, column = img.shape
    noise_salt = np.random.randint(0, 256, (row, column))
    noise_pepper = np.random.randint(0, 256, (row, column))
    rand = 0.1
    noise_salt = np.where(noise_salt < rand * 256, 255, 0)
    noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
    img.astype("float")
    noise_salt.astype("float")
    noise_pepper.astype("float")
    img = img + noise_salt + noise_pepper
    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    img = img.astype("uint8")
    return img

def read_directory_mean(input_directory_name, output_directory_name, mean_image, mean_image_path):
    """
    Read the image in the folder and output the defects image in directory 1 and other raw images in directory 2.

    :param input_directory_name: string. Path of the input folder which contains original images.
    :param output_directory_name: string. Path of the output folder which will contain images after extracting contours.
    :param mean_image: numpy.ndarray. Matrix of the mean image which can be used to replace the rest area.
    """
    count = 0
    for filename in os.listdir(input_directory_name):
        img = find_front_image(input_directory_name + "//" + filename)
        img = replace_image(img, mean_image, mean_image_path)
        cv2.imwrite(output_directory_name + "//" + filename, img)
        print('Finish: ' + filename)
        count = count + 1
        print(count + ' in total.')


dir = 'D:/SteelImage/Final/image_2_defect'
dir_2 = 'D:/SteelImage/Final/image_2_final'
mean_image_path = 'D:/SteelImage/Final/mean_image.jpg'
regex_list = ["/*.jpg"]
mean_image = getMeanImage(dir, regex_list)
#read_directory_mean(dir, dir_2, mean_image, mean_image_path)

input_dir = 'D:/SteelImage/test/new/SIS_00018095_10_0864.jpg'
img = find_front_image(input_dir)
img = replace_image(img, mean_image, mean_image_path)
cv2.imshow('final', img)
