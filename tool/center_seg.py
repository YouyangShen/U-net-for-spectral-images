# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:17:51 2021

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
import maxflow


path = r'../raw data/BA@37/199016_-1_1.hips'

## read the image
image = msi.read(path)
## transfer to nparray
image = image.pixel_values
## save
red_channel = image[:,:,5]


def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])


# %% Inspect the image and the histogram
I = red_channel/(255)

fig, ax = plt.subplots()
ax.imshow(I, cmap=plt.cm.gray)

edges = np.linspace(0, 1, 257)
fig, ax = plt.subplots()
ax.hist(I.ravel(), edges)
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('intensity histogram')
#%% Define likelihood
mu = np.array([0.01, 0.06])
U = np.stack([(I-mu[i])**2 for i in range(len(mu))],axis=2)
S0 = np.argmin(U,axis=2)

fig, ax = plt.subplots()
ax.imshow(S0)
ax.set_title('max likelihood')

#%% Define prior, construct graph, solve
beta  = 0.1
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(I.shape)
g.add_grid_edges(nodeids, beta)
g.add_grid_tedges(nodeids, U[:,:,1], U[:,:,0])

#  solving
g.maxflow()
S = g.get_grid_segments(nodeids)

fig, ax = plt.subplots()
ax.imshow(S)
ax.set_title('max posterior')

fig, ax = plt.subplots()
segmentation_histogram(ax, I, S, edges=edges)
ax.set_aspect(1./ax.get_data_ratio())
ax.set_xlabel('pixel values')
ax.set_ylabel('count')
ax.set_title('segmentation histogram')




# def Dish_label(im,low_threshold = 350, high_threshold = 370):
#     edges = canny(im, sigma=1, low_threshold=10, high_threshold=25)
#     # Detect two radii
#     hough_radii = np.arange(low_threshold, high_threshold, 2)
#     hough_res = hough_circle(edges, hough_radii)
    
#     # Select the most prominent 3 circles
#     accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                                 total_num_peaks=1)
#     return accums, cx, cy, radii

# acc,cx,cy,rad = Dish_label(red_channel)

# # Draw them
# fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
# ax.imshow(red_channel, cmap='gray')

# # for center_y, center_x, radius in zip(cy, cx, rad):
# #     circy, circx = circle_perimeter(center_y, center_x, radius,
# #                                     shape=red_channel.shape)
# #     plt.plot(circx, circy, 'r')
# for i in range(rad.shape[0]):
#     draw_circle = plt.Circle((cx[i], cy[i]), rad[i],fill=False,color='r')
#     plt.gcf().gca().add_artist(draw_circle)
# plt.show()