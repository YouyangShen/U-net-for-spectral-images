# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:50:42 2021

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

path = r'BA@37/232473_-1_1.hips'

## read the image
image = msi.read(path)
## transfer to nparray
image = image.pixel_values
## save
red_channel = image[:,:,5]
plt.imshow(red_channel,cmap='gray')
# red_channel = cv.normalize(red_channel,None)
gray = cv.medianBlur(red_channel, 3)
gray = np.uint8(gray)
# plt.imshow(gray)
edges = cv.Canny(gray, 5, 10)
# plt.imshow(edges)



rows = gray.shape[0]
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, rows /16,
                            param1=100, param2=25,
                            minRadius=340, maxRadius=370)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(gray, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(gray, center, radius, (255, 0, 0), 3)

plt.imshow(gray,cmap='gray')

[n,m,r] = np.shape(image)
mask = np.zeros((n,m,r))

for i in range(n):
    for j in range(m):
        if np.sqrt((center[0] - j)**2 + (center[1]-i)**2)<=radius:
            mask[i,j,:]=image[i,j,:]
        else: 
            mask[i,j,:]=0
    
# msi.write(mask,'232473_-1_1.hips')





# plt.imshow(gray)
# cv.imshow("detected circles", red_channel)
# cv.waitKey(0)
