# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:59:26 2021

@author: Youyang Shen
"""

import numpy as np
import cv2 as cv
import os
import skimage.io
from matplotlib import pyplot as plt
# import images

path = r'C:\Users/Youyang Shen/Documents/Videometer/VideometerLab/Session/Results/TC/TC 0001/'

# filename = '149521_0_1_out.png'
for filename in os.listdir(path):
    print(filename)
    image = cv.imread(path+filename)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap = 'gray')
    N,M = gray.shape
    
    max_p = np.max(gray)
    
    gray_mask = gray>0
    gray_mask = np.array(gray_mask,dtype=int)*255
    fig, ax = plt.subplots()
    plt.imshow(gray_mask, cmap='gray')
    plt.show()
    
    cv.imwrite(filename,gray_mask)