# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 23:47:24 2022

@author: Youyang Shen
"""

import numpy as np
from matplotlib import pyplot as plt
import tifffile as tf
import os
import cv2
import PIL

img_path = r'NN/Pytorch-UNet/data/PCA/'
mask_path = r'NN/Pytorch-UNet/data/open_masks/'
img_store_path = r'NN/Pytorch-UNet/data/PCA_aug/'
mask_store_path = r'NN/Pytorch-UNet/data/PCA_mask_aug/'

def random_crop(img, mask, crop_size=(1000, 1000)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    mask = mask[y:y+crop_size[0], x:x+crop_size[1]]
    img = cv2.resize(img, (w,h))
    mask = cv2.resize(mask, (w,h))
    return img,mask

def h_flip(image,mask):
    return  np.fliplr(image),np.fliplr(mask)

def v_flip(image,mask):
    return np.flipud(image),np.flipud(mask)

def blur_image(image,mask):
    return cv2.GaussianBlur(image, (9,9),0),cv2.GaussianBlur(mask, (9,9),0)

for filename in os.listdir(img_path):
    image = tf.imread(img_path+filename)
    print(filename)
    mask_name = filename.split(".")[0] + '_mask.png'
    mask = cv2.imread(mask_path+mask_name,0)
    
    new_image,new_mask = v_flip(image,mask)

    new_filename = 'v'+filename
    new_maskname = 'v'+mask_name

    
    tf.imwrite(img_store_path+new_filename,new_image)
    cv2.imwrite(mask_store_path+new_maskname,new_mask)


