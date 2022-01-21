# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:51:17 2021

@author: Youyang Shen
"""

import tifffile
import msi
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
import cv2
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import color

path = r'../raw data/BA@37/199016_-1_1.hips'

## read the image
image = msi.read(path)
## transfer to nparray
image = image.pixel_values
## save
red_channel = image[:,:,5]

#%% function
def getGaussDerivative(t):
    '''
    Computes kernels of Gaussian and its derivatives.
    Parameters
    ----------
    t : float
        Vairance - t.

    Returns
    -------
    g : numpy array
        Gaussian.
    dg : numpy array
        First order derivative of Gaussian.
    ddg : numpy array
        Second order derivative of Gaussian
    dddg : numpy array
        Third order derivative of Gaussian.

    '''

    kSize = 5
    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    g = g/np.sum(g)
    dg = -x/t*g
    ddg = -g/t - x/t*dg
    dddg = -2*dg/t - x/t*ddg
    return g, dg, ddg, dddg

# Show circles
def getCircles(coord, scale):
    '''
    Comptue circle coordinages

    Parameters
    ----------
    coord : numpy array
        2D array of coordinates.
    scale : numpy array
        scale of individual blob (t).

    Returns
    -------
    circ_x : numpy array
        x coordinates of circle. Each column is one circle.
    circ_y : numpy array
        y coordinates of circle. Each column is one circle.

    '''
    theta = np.arange(0, 2*np.pi, step=np.pi/100)
    theta = np.append(theta, 0)
    circ = np.array((np.cos(theta),np.sin(theta)))
    n = coord.shape[0]
    m = circ.shape[1]
    circ_y = np.sqrt(2*scale)*circ[[0],:].T*np.ones((1,n)) + np.ones((m,1))*coord[:,[0]].T
    circ_x = np.sqrt(2*scale)*circ[[1],:].T*np.ones((1,n)) + np.ones((m,1))*coord[:,[1]].T
    return circ_x, circ_y

# %% Set parameters
def detectFibers(im, diameterLimit, stepSize, tCenter, thresMagnitude):
    '''
    Detects fibers in images by finding maxima of Gaussian smoothed image

    Parameters
    ----------
    im : numpy array
        Image.
    diameterLimit : numpy array
        2 x 1 vector of limits of diameters of the fibers (in pixels).
    stepSize : float
        step size in pixels.
    tCenter : float
        Scale of the Gaussian for center detection.
    thresMagnitude : float
        Threshold on blob magnitude.

    Returns
    -------
    coord : numpy array
        n x 2 array of coordinates with row and column coordinates in each column.
    scale : numpy array
        n x 1 array of scales t (variance of the Gaussian).

    '''
    
    radiusLimit = diameterLimit/2
    radiusSteps = np.arange(radiusLimit[0], radiusLimit[1]+0.1, stepSize)
    tStep = radiusSteps**2/np.sqrt(2)
    
    r,c = im.shape
    n = tStep.shape[0]
    L_blob_vol = np.zeros((r,c,n))
    for i in range(0,n):
        g, dg, ddg, dddg = getGaussDerivative(tStep[i])
        L_blob_vol[:,:,i] = tStep[i]*(cv2.filter2D(cv2.filter2D(im,-1,g),-1,ddg.T) + 
                                      cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,g.T))
    # Detect fibre centers
    g, dg, ddg, dddg = getGaussDerivative(tCenter)
    Lg = cv2.filter2D(cv2.filter2D(im, -1, g), -1, g.T)
    
    coord = skimage.feature.peak_local_max(Lg, threshold_abs = thresMagnitude)
    
    # Find coordinates and size (scale) of fibres
    magnitudeIm = np.min(L_blob_vol, axis = 2)
    scaleIm = np.argmin(L_blob_vol, axis = 2)
    scales = scaleIm[coord[:,0], coord[:,1]]
    magnitudes = -magnitudeIm[coord[:,0], coord[:,1]]
    idx = np.where(magnitudes > thresMagnitude)
    coord = coord[idx[0],:]
    scale = tStep[scales[idx[0]]]
    return coord, scale

def Dish_label(im,low_threshold = 340, high_threshold = 370):
    edges = canny(im, sigma=1, low_threshold=5, high_threshold=10)
    # Detect two radii
    hough_radii = np.arange(low_threshold, high_threshold, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    return accums, cx, cy, radii
#%%
# Radius limit
diameterLimit = np.array([340,370])
stepSize = 0.3

# Parameter for Gaussian to detect center point
tCenter = 20

# Parameter for finding maxima over Laplacian in scale-space
thresMagnitude = 8

acc,cx,cy,rad = Dish_label(red_channel)
# Detect fibres
coord, scale = detectFibers(red_channel, diameterLimit, stepSize, tCenter, thresMagnitude)

mask = []
mask_r = []
for i in range(coord.shape[0]):
    if np.sqrt((cx - coord[i][1])**2 + (cy-coord[i][0])**2)<=rad:
        mask.append(coord[i])
        mask_r.append(scale[i])
        
mask = np.array(mask)
mask_r = np.array(mask_r)

#plot dish

# Draw them
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(red_channel, cmap='gray')

# for center_y, center_x, radius in zip(cy, cx, rad):
#     circy, circx = circle_perimeter(center_y, center_x, radius,
#                                     shape=red_channel.shape)
#     plt.plot(circx, circy, 'r')

draw_circle = plt.Circle((cx, cy), rad,fill=False,color='r')
plt.gcf().gca().add_artist(draw_circle)
plt.show()

# Plot detected fibres
fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(red_channel, cmap='gray')
ax.plot(mask[:,1], mask[:,0], 'r.')

circ_x, circ_y = getCircles(mask, mask_r)
plt.plot(circ_x, circ_y, 'r')