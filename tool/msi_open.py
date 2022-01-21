# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:30:29 2021

@author: Youyang Shen
"""
import tifffile
import msi

path = r'C:\Users\Youyang Shen\OneDrive - Danmarks Tekniske Universitet\Study\special course\Images\1_20170410_151511_241.hips'

## read the image
image = msi.read(path)
## transfer to nparray
image = image.pixel_values
## save
tifffile.imsave("test.tiff",image)

path2 = r'..\Images\1_20170410_151511_241_Simple Threshold.hips'

seg = msi.read(path2)
seg = seg.pixel_values


