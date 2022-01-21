# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:12:50 2021

@author: Youyang Shen
"""

import PIL
import os
import numpy as np
import torch
import tifffile

path = r'BA@37/'
path2 = r'aug result dice/'
# open gt
gt_path = path+'open/'
vm_path = path+'vm_open/'
gt_list = []
vm_list = []
unet_list =[]
for filename in os.listdir(gt_path):
    gt = PIL.Image.open(gt_path+filename)
    vm_name = filename.split(".")[0] + '_out.png'
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

gt_list = torch.Tensor(gt_list) 
vm_list = torch.Tensor(vm_list)
unet_list = torch.Tensor(unet_list)

EPS = 1e-6
#slightly modified
def get_IoU(outputs, labels):
    outputs = outputs.int()
    labels = labels.int()
    print(outputs.shape)
    print(labels.shape)
    # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    return iou



IoU=get_IoU(unet_list,gt_list)

print(path,IoU.mean())
# intersection = (vm_list[0].int() & gt_list[0].int()).float()
# intersection.shape