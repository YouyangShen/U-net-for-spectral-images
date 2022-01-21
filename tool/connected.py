# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 12:06:00 2021

@author: Youyang Shen
"""
import os
from math import sqrt
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

import PIL
import torch
import tifffile
from statistics import mean

# path_name = ['BA@37/','PCA@22/','PCA@37/','PCA@37 with counting/']

path_name = ['BA@37/']

for i in range(len(path_name)):
    path = path_name[i]
    path2 = r'small result/'
    # open gt
    gt_path = path+'open/'
    vm_path = path+'vm_open/'
    gt_list = []
    vm_list = []
    unet_list =[]
    for filename in os.listdir(gt_path):
        gt = PIL.Image.open(gt_path+filename)
        vm_name = filename.split(".")[0] + '_out.png'
        # print(vm_name)
        unet_name = filename.split(".")[0]+ '.tiff'
        vm = PIL.Image.open(vm_path+vm_name)
        unet = tifffile.imread(path2+unet_name)
        unet = unet/127*255
        gt = np.asarray(gt)
        # print(np.unique(gt))
        vm = np.asarray(vm)
        unet = np.asarray(unet)
        # print(np.unique(vm))
        # gt = torch.Tensor(gt)
        # vm = torch.Tensor(vm)
        gt_list.append(gt)
        vm_list.append(vm)
        unet_list.append(unet)
    
    # path = 'NN/Pytorch-UNet/data/masks/87537_0_1_mask.png'
    # path2 = 'small result/87537_0_1.tiff'
    def get_pre_and_recall(image1,image2):
    # image = cv.imread(path,0)
    # image2 = cv.imread(path2,0)
    # plt.imshow(image)
    
        mask_out = cv.connectedComponentsWithStats(image1)
        image2 = np.uint8(image2)
        predict_out = cv.connectedComponentsWithStats(image2)
        
        
        count = mask_out[0]-1
        
        mask_center = mask_out[3].astype(int)
        mask_center = mask_center[1:,:]
        count2 = predict_out[0]-1
        
        predict_center = predict_out[3].astype(int)
        predict_center=predict_center[1:,:]
        tp = []
        
        for i in range(count2):
            tp.append(image1[predict_center[i,1],predict_center[i,0]])
        
        tp = np.asarray(tp)    
        tp_nr = len(tp[tp[:]==255])
        
        fp_nr = count2-tp_nr
        
        fn = []
        
        for i in range(count):
            fn.append(image2[mask_center[i,1],mask_center[i,0]])
        
        fn = np.asarray(fn)    
        fn_nr = len(fn[fn[:]==0])
        EPS = 1e-4
        precision = (tp_nr+EPS)/(tp_nr+fp_nr+EPS)
        
        recall = (tp_nr+EPS)/(tp_nr+fn_nr+EPS)
        
        error = abs(count-count2)
        
        
        return precision,recall,error
    
    
    precision =[]
    recall = []
    error =[]
    for i in range(len(gt_list)):
        pre,rec,err = get_pre_and_recall(gt_list[i],unet_list[i])
        precision.append(pre)
        recall.append(rec)
        error.append(err)
    mean_precision = mean(precision)
    mean_recall = mean(recall)
    mean_error = mean(error)
    
    print(path,mean_precision,mean_recall,mean_error)