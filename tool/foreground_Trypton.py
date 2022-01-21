# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:48:17 2021

@author: Youyang Shen
"""

import sys
import cv2 as cv
import numpy as np
import tifffile
import msi
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import color
import os
import matplotlib.cm as cm

path = r'../raw data/TryptonEC@37/'


for filename in os.listdir(path):
    print(filename)
    image = msi.read(path+filename)
    image = image.pixel_values
    red_channel = image[:,:,5]

    [n,m,r] = np.shape(image)
    mask = np.zeros((n,m,r))
    gray = cv.medianBlur(red_channel, 3)
    gray = np.uint8(gray)
    # plt.imshow(gray)
    edges = cv.Canny(gray, 5, 10)
    # plt.imshow(edges)
    rows = gray.shape[0]
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, rows /16,
                                param1=100, param2=50,
                                minRadius=400, maxRadius=410)
    center = []
    radius = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
    for i in range(n):
        for j in range(m):
            if np.sqrt((center[0] - j)**2 + (center[1]-i)**2)<=radius-17:
                mask[i,j,:]=image[i,j,:]
            else: 
                mask[i,j,:]=0
        
    msi.write(mask,filename)


# plt.imshow(gray)
# cv.imshow("detected circles", red_channel)
# cv.waitKey(0)
