# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:12:32 2021

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


path = r'TryptonTC/'


for filename in os.listdir(path):
    print(filename)
    image = msi.read(path+filename)
    image = image.pixel_values
    suffix = filename.split(".")[0]
    print(suffix)
    filename = suffix+'.tiff'
    # tifffile.imsave(filename,image)


# plt.imshow(gray)
# cv.imshow("detected circles", red_channel)
# cv.waitKey(0)
