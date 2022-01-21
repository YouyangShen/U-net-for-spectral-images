# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:45:49 2022

@author: Youyang Shen
"""

import cv2 as cv
import numpy as np
import os
import PIL
from matplotlib import pyplot as plt


# path_name = ['BA@37/','PCA@22/','PCA@37/','PCA@37 with counting/','TryptonEC@37/','TryptonTC/']

path_name = ['NN/Pytorch-UNet/data/']

for i in range(len(path_name)):
    path = path_name[i]
    gt_path = path+'masks/'
    open_path = path+'open_masks/'
    

    kernel = np.ones((3,3),np.uint8)
    for filename in os.listdir(gt_path):
        # gt = PIL.Image.open(gt_path+filename)
    
        # path = r'PCA@37/gt/149522_0_1.png'
        gt =cv.imread(gt_path+filename,0)
        # gt_1 = PIL.Image.open(gt_path+filename)
        gt_open = cv.morphologyEx(gt, cv.MORPH_OPEN, kernel)
        gt_open = gt_open/255
        cv.imwrite(os.path.join(open_path,filename),gt_open)
# plt.imshow(gt)
# plt.imshow(gt_open)
    
