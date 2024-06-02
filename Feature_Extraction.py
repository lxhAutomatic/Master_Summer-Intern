# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:10:42 2022
@author: Xinhao Lan
"""
# built-in modules
import glob

# third-party modules
import torchvision.models as models
import torch
import numpy as np
import cv2

def read_image(input_dir):
    """
    Read the image in the dataset from the direcatory called input_dir

    :param input_dir: string. The inpur direcatory which includes the dataset folder
    
    :return images: list. The type of each item in the list is numpy.ndarray. Each one is a matrix of the image.
    :return paths: list. The type of each item in the list is string. Each one is the direcatory of the image.
    """
    glob_dir = input_dir + "/*.jpg"
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)] # get the images and use the resize funcrion
    paths = [file for file in glob.glob(glob_dir)] # get the path for the images file
    return images, paths

def get_features(images, i, model_path):
    """
    Use Different pretrained model which are pretrained on the Imagenet to get the feature.
    
    Comment all other networks to make sure there is only one pretrained model.
    
    :param images: list. The type of each item in the list is numpy.ndarray. Each one is a matrix of the original image.

    :return pred_images: list. The type of each item in the list is numpy.ndarray. Each one is a matrix of the feature after the pretrained model.
    """
    images = images[20*i: 20*(i+1)]
    images = np.array(np.float32(images).reshape(len(images), -1) / 255)
    # model = models.mobilenet_v3_large(pretrained=True) #Mobilenet_v3_large
    # model = models.resnet34(pretrained=True) #Resnet_34
    # model = models.densenet161(pretrained=True) #Densenet_161
    # model = models.vit_l_32(pretrained=True) #VisionTransformer
    # model = models.shufflenet_v2_x1_0(pretrained=True) #Shufflenet_v2
    model = models.vit_l_32(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    predictions = model(torch.from_numpy(images.reshape(-1, 3, 224, 224)))
    pred_images = predictions.reshape(images.shape[0], -1)
    pred_images = pred_images.detach().numpy()
    return pred_images

def extract_feature(images, output_dir, model_path):
    """
    Function to use the pretrained model to get the feature.
    
    Since the RAM is limited, use the loop to extract 20 image feature one time. 

    :param images : list. The type of each item in the list is numpy.ndarray. Each one is a matrix of the image.
    :output_dir: string. The output file name for the feature file.
    """
    pred_images = np.empty((0,1000))
    print(len(images))
    
    for i in range (0, int(len(images)/20)):
        pred_image = get_features(images, i, model_path)
        print(pred_image.shape)
        pred_images = np.vstack((pred_images,pred_image))
    print(pred_images.shape)
    np.savetxt(output_dir, pred_images, fmt = "%f") 
    print('done')

input_dir = 'D:/SteelImage/Final/image_2_final'
output_dir = 'D:/SteelImage/Final/Transformer_feature_v2'
model_path = 'C:/Users/75581/Desktop/Summer/Delievered/vit_l_32-c7638314.pth'
images, paths = read_image(input_dir)
extract_feature(images, output_dir, model_path)


